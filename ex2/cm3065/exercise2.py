from pathlib import Path
from scipy.io import wavfile
import os
import soundfile as sf

def linear_predict_order_2_encode(samples: list[int]) -> list[int]:
    """Linear prediction: predict as 2*prev - prev_prev"""
    if len(samples) < 2:
        return samples[:]
    
    residuals = samples[:2]  # Store first two as-is
    for i in range(2, len(samples)):
        predicted = 2 * samples[i-1] - samples[i-2]
        residual = samples[i] - predicted
        residuals.append(residual)
    return residuals

def linear_predict_order_2_decode(residuals: list[int]) -> list[int]:
    """Decode 2nd order linear prediction residuals"""
    if len(residuals) < 2:
        return residuals[:]
    
    samples = residuals[:2]  # First two are stored as-is
    for i in range(2, len(residuals)):
        predicted = 2 * samples[i-1] - samples[i-2]
        sample = residuals[i] + predicted
        samples.append(sample)
    return samples

def rice_decode_all(bits: str, k: int) -> list[int]:
    m = 2 ** k
    results = []
    idx = 0
    n = len(bits)
    
    while idx < n:
        # Find the unary code: number of 1s before the first 0 starting at idx
        q = 0
        while idx + q < n and bits[idx + q] == '1':
            q += 1
        
        # The next bit must be zero (delimiter)
        if idx + q >= n or bits[idx + q] != '0':
            raise ValueError("Invalid encoding: unary code not properly terminated")
        
        # Start of remainder bits
        start_r = idx + q + 1
        end_r = start_r + k
        
        if end_r > n:
            raise ValueError("Invalid encoding: remainder bits incomplete")
        
        r_bits = bits[start_r:end_r]
        r = int(r_bits, 2) if r_bits else 0
        
        s = q * m + r
        results.append(s)
        
        # Move index past this integer's bits: unary(q 1's + 0) + k remainder bits
        idx = end_r
    
    return results

def rice_encode(s: int, k: int) -> str:
    bits = []
    
    # unary code for q (q times 1 then 0)
    q = s >> k  # quotient
    bits.extend(([1] * q) + [0])
    
    m = 2 ** k
    r = s & (m - 1) # remainder: last k bits
    for i in range(k-1, -1, -1):
        bits.append((r >> i) & 1)
        
    return ''.join(map(str, bits))

def zigzag_encode(n: int) -> int:
    # For positive numbers, the representation is the number doubled,
    # for negative numbers, the representation is the number multiplied by -2 and has 1 subtracted.
    return (n << 1) ^ (n >> 31)

def zigzag_decode(n: int) -> int:
    return (n >> 1) ^ -(n & 1)

def file_size_kb_str(path: Path | str):
    return f"{os.path.getsize(path) / 1024:.2f} KB"

def write_bitstring_to_file_with_padding_count(bit_str: str, filename: str) -> None:
    # Calculate padding needed to make length multiple of 8
    padding = (8 - len(bit_str) % 8) % 8  # 0 to 7
    padded_bit_str = bit_str.ljust(len(bit_str) + padding, '0')
    
    byte_array = bytearray()
    # First byte is the padding count
    byte_array.append(padding)
    
    # Convert every 8 bits to a byte
    for i in range(0, len(padded_bit_str), 8):
        byte_chunk = padded_bit_str[i:i+8]
        byte_val = int(byte_chunk, 2)
        byte_array.append(byte_val)
    
    with open(filename, 'wb') as f:
        f.write(byte_array)

def read_bitstring_from_file_with_padding_count(filename: str) -> str:
    with open(filename, 'rb') as f:
        data = f.read()
    padding = data[0]  # first byte is padding count
    byte_data = data[1:]
    bit_str = ''.join(f'{byte:08b}' for byte in byte_data)
    if padding > 0:
        bit_str = bit_str[:-padding]  # remove padding bits at the end
    return bit_str

if __name__ == "__main__":
    
    # paths = [
    #     Path("/home/f1/projects/cm3065_final/files/Exercise2_Files/Sound1.wav"),
    #     Path("/home/f1/projects/cm3065_final/files/Exercise2_Files/Sound2.wav"),
    # ]
    # print(f"sound1 size: {file_size_kb_str(paths[0])}")
    # print(f"sound2 size: {file_size_kb_str(paths[1])}")


    base_filename = "Sound1" 
    input_wav_path = f"{base_filename}.wav" 
    encoded_path = f"{base_filename}_Enc.ex2"
    # source_audio_data, source_sr = sf.read(input_wav_path, dtype='int32')
    # source_audio_data = source_audio_data.astype(int).tolist()
    # print(source_audio_data)
    
    # path = paths[0]
    k = 4
    sr, expected = wavfile.read(input_wav_path)
    source_audio_data: list[int] = list(map(int, expected))
    #
    print("Encoding")
    residuals: list[int] = linear_predict_order_2_encode(source_audio_data)
    unsigned: list[int] = list(map(zigzag_encode, residuals))
    riced: str = ''.join(rice_encode(i, k) for i in unsigned)
    
    # out_path = path.parent / (path.stem + "_Enc.ex2")
    write_bitstring_to_file_with_padding_count(riced, encoded_path)
    print(f"{encoded_path} (K = {k} bits): {file_size_kb_str(encoded_path)}")
    riced: str = read_bitstring_from_file_with_padding_count(encoded_path)
    
    unsigned: list[int] = rice_decode_all(riced, k)
    residuals: list[int] = list(map(zigzag_decode, unsigned))
    decoded: list[int] = linear_predict_order_2_decode(residuals)
    assert decoded == source_audio_data
