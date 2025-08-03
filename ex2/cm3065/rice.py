import numpy as np
import soundfile as sf
import wave
from bitarray import bitarray
from bitarray.util import int2ba, ba2int
import json
import os # For path manipulation and checking file existence
import time # For time tracking

# --- Core Rice Coding Functions ---

def write_residuals(bit_array: bitarray, filename: str) -> None:
    """
    Writes a bitarray to a file, prepending a byte that specifies the padding count.

    The bitarray is padded with '0's to a multiple of 8 bits before writing.
    The number of padding bits (0-7) is stored in the first byte of the file,
    followed by the padded bitstream.

    Args:
        bit_array (bitarray): The bitarray object to write.
        filename (str): The path to the output file.
    """
    # Create a copy to avoid modifying the original bitarray
    padded_bit_array = bit_array.copy()

    # Calculate padding needed to make length a multiple of 8
    padding_count = (8 - len(padded_bit_array) % 8) % 8  # Value will be 0 to 7
    
    # Extend the bitarray with '0's for padding
    padded_bit_array.extend('0' * padding_count)
    
    # Convert the padded bitarray to a bytes object
    data_bytes = padded_bit_array.tobytes()
    
    # Prepend the padding count as the first byte
    output_bytes = bytes([padding_count]) + data_bytes
    
    with open(filename, 'wb') as f:
        f.write(output_bytes)
    
    print(f"Wrote {len(bit_array)} original bits with {padding_count} padding bits to '{filename}'.")

def read_residuals(filename: str) -> bitarray:
    """
    Reads a bitstream from a file that was written using
    write_bitarray_to_file_with_padding_count and returns the original bitarray.

    Args:
        filename (str): The path to the input file.

    Returns:
        bitarray: The original, unpadded bitarray object.
    """
    with open(filename, 'rb') as f:
        data = f.read()

    if not data:
        return bitarray() # Return empty bitarray for an empty file

    # The first byte is the padding count
    padding_count = data[0]
    
    # The rest of the bytes are the encoded data
    bitstream_bytes = data[1:]
    
    # Convert the bytes back to a bitarray
    read_bit_array = bitarray()
    read_bit_array.frombytes(bitstream_bytes)
    
    # Slice the bitarray to remove the padding bits
    original_length = len(read_bit_array) - padding_count
    unpadded_bit_array = read_bit_array[:original_length]
    
    print(f"Read {len(unpadded_bit_array)} original bits from '{filename}'.")

    return unpadded_bit_array
def encode_high_order_predictor(signal):
    """
    Generates residuals using a hardcoded 4th-order linear predictor.
    The first 4 samples are treated as unpredicted.
    
    The 4th-order predictor formula is:
    predicted_sample[n] = 4*signal[n-1] - 6*signal[n-2] + 4*signal[n-3] - signal[n-4]
    """
    # The predictor order is now a fixed value
    predictor_order = 4
    
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal, dtype=np.int32)
    
    if len(signal) <= predictor_order:
        return signal # Return original signal if too short for prediction

    residuals = np.zeros_like(signal, dtype=np.int32)

    # The initial 4 samples are their own "residuals" (or unpredicted values)
    residuals[:predictor_order] = signal[:predictor_order]
    
    # Apply the hardcoded 4th-order predictor
    for n in range(predictor_order, len(signal)):
        # Calculate the predicted sample using the 4th-order formula
        predicted_sample = (4 * signal[n-1] - 6 * signal[n-2] + 4 * signal[n-3] - signal[n-4])
        residuals[n] = signal[n] - predicted_sample
            
    return residuals

def decode_high_order_predictor(residuals):
    """
    Reconstructs the original signal from residuals using a hardcoded 4th-order linear predictor.
    The first 'predictor_order' samples are treated as unpredicted and are direct signal values.
    """
    predictor_order = 4
    if not isinstance(residuals, np.ndarray):
        residuals = np.array(residuals, dtype=np.int32)

    if len(residuals) <= predictor_order:
        return residuals # If too short, residuals are the signal itself

    reconstructed_signal = np.zeros_like(residuals, dtype=np.int32)

    # The initial 'predictor_order' samples are directly the residuals
    reconstructed_signal[:predictor_order] = residuals[:predictor_order]

    # Reconstruct the signal using the inverse of the 4th-order predictor
    for n in range(predictor_order, len(residuals)):
        predicted_sample_reconstructed = (
            4 * reconstructed_signal[n-1]
            - 6 * reconstructed_signal[n-2]
            + 4 * reconstructed_signal[n-3]
            - reconstructed_signal[n-4]
        )
        reconstructed_signal[n] = residuals[n] + predicted_sample_reconstructed
        
    return reconstructed_signal

def rice_encode(samples, k):
    """
    Rice encodes an array of integers 'samples' (residuals) with parameter 'k'
    and returns their combined Rice code as a single bitarray.
    Handles signed integers using folding (interleaving).
    """
    encoded_bitstream = bitarray()
    for s in samples:
        if s < 0:
            s_folded = (abs(s) << 1) - 1
        else:
            s_folded = s << 1

        m = 2 ** k
        quotient = s_folded >> k
        remainder = s_folded & (m - 1)

        unary_bits = bitarray('1' * quotient + '0')

        remainder_bits = int2ba(int(remainder), length=k)

        encoded_bitstream.extend(unary_bits + remainder_bits)
    return encoded_bitstream


def rice_decode(bit_stream: bitarray, k: int) -> list[int]:
    """
    Decodes all Rice-encoded numbers from a bitarray.
    Returns a list of decoded samples.
    Optimized for performance by using bitarray.index() for unary decoding.
    """
    decoded_samples = []
    current_bit_offset = 0
    len_bit_stream = len(bit_stream) # Store length to avoid repeated calls

    # Pre-calculate 2^k as it's used repeatedly
    two_pow_k = 1 << k

    while current_bit_offset < len_bit_stream:
        # Progress indicator (optional, remove for max performance)
        percentage = (current_bit_offset / len_bit_stream) * 100
        print(f"Progress: {percentage:.2f}%")

        try:
            # Find the '0' terminator for the unary part
            # This directly gives us the end of the unary sequence (and thus the quotient)
            zero_terminator_index = bit_stream.index(False, current_bit_offset)
        except ValueError:
            # If no '0' is found, it means the stream is malformed or ends with '1's.
            # If we are at the very beginning and no '0' (meaning empty stream), break.
            if current_bit_offset == len_bit_stream:
                break # End of stream, no more numbers
            else:
                raise ValueError(f"Malformed Rice code: No '0' terminator found after bit offset {current_bit_offset}. Stream ended prematurely or malformed.")
        
        # Quotient is the number of '1's before the '0'
        quotient = zero_terminator_index - current_bit_offset
        current_bit_offset = zero_terminator_index + 1 # Move past the '0' terminator

        # Extract R (remainder) - Binary part of k bits
        remainder_start = current_bit_offset
        remainder_end = current_bit_offset + k

        if remainder_end > len_bit_stream:
            raise ValueError(f"Malformed Rice code: Not enough bits for remainder (expected {k}, available {len_bit_stream - current_bit_offset}) at bit offset {remainder_start}.")

        # Convert remainder bits to integer
        remainder = ba2int(bit_stream[remainder_start:remainder_end])
        current_bit_offset = remainder_end # Update offset to after remainder

        # Reconstruct folded value: s_folded = Q * 2^k + R
        s_folded = (quotient * two_pow_k) + remainder

        # Unfold the value back to signed integer (signed Golomb-Rice coding)
        if s_folded & 1 == 0:  # Check if even using bitwise AND
            s_unfolded = s_folded >> 1
        else:
            s_unfolded = -((s_folded + 1) >> 1)
        
        decoded_samples.append(s_unfolded)
            
    return decoded_samples

if __name__ == "__main__":
    base_filename = "Sound2" 
    input_wav_path = f"{base_filename}.wav" 
    encoded_path = f"{base_filename}_Enc.ex2"
    
    K = 4

    source_audio_data, source_sr = sf.read(input_wav_path, dtype='int32')
    residuals = encode_high_order_predictor(source_audio_data)
    print(residuals)
    print("Encoding...")
    residuals = rice_encode(residuals, K)
    write_residuals(residuals, encoded_path)


    residuals = read_residuals(encoded_path)
    print("Decoding...")
    decoded_residuals = rice_decode(residuals, K)
    reconstructed_signal = decode_high_order_predictor(decoded_residuals)
    sf.write(f"{base_filename}_EncDec.wav", reconstructed_signal, source_sr)

    roundtrip_audio_data, _ = sf.read(f"{base_filename}_EncDec.wav", dtype='int32')
    assert np.array_equal(source_audio_data, roundtrip_audio_data)

