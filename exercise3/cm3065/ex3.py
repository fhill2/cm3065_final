import ffmpeg
import subprocess
import json
import os
from pathlib import Path

def encode_video(input_path: Path, params):
    args = dict(
        vcodec=params["video_codec"],
        acodec=params["audio_codec"],
        r=params["frame_rate"],
        aspect=params['aspect_ratio'],
        # ffmpeg-python expects 'k' suffix for kilobits
        video_bitrate = f"{params['video_bit_rate']['max']}k",
        audio_bitrate = f"{params['audio_bit_rate']['max']}k",
        ac=params['audio_channels']

    )
    stream = ffmpeg.input(input_path)

    res_parts = str(params['resolution']).split('x')
    width = res_parts[0].strip()
    height = res_parts[1].strip()

    stream = stream.video.filter('scale', width=width, height=height)

    combined_stream = ffmpeg.concat(stream, ffmpeg.input(input_path).audio, v=1, a=1)

    output_path = input_path.parent / Path(str(input_path.stem) + "_formatOK.mp4")

    (
        combined_stream
        .output(str(output_path), **args)
        .run(overwrite_output=True)
    )
    print(f"Video encoded successfully to: {output_path}")

def parse_aspect(width, height):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    common_divisor = gcd(width, height)
    return f"{width // common_divisor}:{height // common_divisor}"

def parse_frame_rate(frame_rate):
    if '/' in frame_rate:
        num, den = map(int, frame_rate.split('/'))
        return round(num / den, 2)

def metadata(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return None

    command = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=True)

    # Parse the JSON output
    metadata = json.loads(result.stdout)

    # Extract format information
    format_info = metadata.get('format', {})
    audio = next((s for s in metadata.get("streams") if s.get("codec_type") == "audio"))
    video = next((s for s in metadata.get("streams") if s.get("codec_type") == "video"))
    return dict(
            video_path = video_path,
            video_format = format_info.get('format_name'),
            video_codec = video["codec_name"],
            audio_codec = audio["codec_name"],
            frame_rate = parse_frame_rate(video["avg_frame_rate"]),
            aspect_ratio = parse_aspect(video["width"], video["height"]),
            resolution = f"{video['width']} x {video['height']}",
            video_bit_rate_mbps = int(video['bit_rate'])  / 1_000_000,
            video_bit_rate = int(video["bit_rate"]),
            audio_bit_rate = int(audio["bit_rate"]),
            audio_channels = audio.get('channels')
    )
if __name__ == "__main__":

    paths = [
        "Cosmos_War_of_the_Planets.mp4",
        "Last_man_on_earth_1964.mov",
        "The_Gun_and_the_Pulpit.avi",
        "The_Hill_Gang_Rides_Again.mp4",
        "Voyage_to_the_Planet_of_Prehistoric_Women.mp4"
    ]

    expected = dict(
        video_codec = "h264",
        audio_codec = "aac",
        frame_rate = 25,
        aspect_ratio = "16:9",
        resolution="640 x 360",
        video_bit_rate=dict(
            min=2_000_000,
            max=5_000_000,
        ),
        audio_bit_rate=dict(
            max=256000
        ),
        audio_channels=2
    )

    info = []

    for path in paths:
        path = Path(f"video/{path}")
        meta = metadata(path)
        requirements = dict(
            video_codec=meta["video_codec"] == expected['video_codec'],
            audio_codec=meta["audio_codec"] == expected["audio_codec"],
            frame_rate=meta["frame_rate"] == expected["frame_rate"],
            aspect_ratio=meta["aspect_ratio"] == expected["aspect_ratio"],
            resolution=meta["resolution"] == expected["resolution"],
            # ffprobe outputs bit rates in bits per second
            # 1 Mb/s = 1_000_000 bits per second
            # 1 kb/s = 1000 bits per second
            video_bit_rate=expected['video_bit_rate']['min'] <= meta["video_bit_rate"] <= expected['video_bit_rate']['max'],
            audio_bit_rate=meta["audio_bit_rate"] <= expected['audio_bit_rate']['max'],
            audio_channels=meta["audio_channels"] == expected['audio_channels']
        )

        report = str(meta['video_path']) +  "\n"
        for key, req in requirements.items():
            report += f"  · {key}: {meta[key]}"
            if req:
                report += " OK"
            else:
                if key == "video_bit_rate":
                    report += f" FAIL, {expected[key]['min']} >= video_bit_rate < {expected[key]['max']}"
                elif key == "audio_bit_rate":
                    report += f" FAIL, < {expected[key]['max']}"
                else:
                    report += f" FAIL, {expected[key]}"

            report += "\n"

        info.append((meta, requirements, report))



    print(f"-------------- REPORTS ---------------")
    for _, _, report in info:
        print(report)
    print(f"-------------- END REPORT ---------------")

    for meta, req, _ in info:
        print(f"-------------- Processing Video {str(meta['video_path'])} ---------------")
        if any(not r for r in req.values()):
            encode_video(meta["video_path"], expected)
 
