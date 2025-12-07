#!/usr/bin/env python3
"""
Stack multiple videos into a grid layout (side-by-side or 2x2, etc.)

Usage:
    # Side by side (1x2)
    python stack_videos.py video1.mp4 video2.mp4 -o output.mp4
    
    # 2x2 grid
    python stack_videos.py video1.mp4 video2.mp4 video3.mp4 video4.mp4 -o output.mp4 --grid 2x2
    
    # Vertical stack (2x1)
    python stack_videos.py video1.mp4 video2.mp4 -o output.mp4 --grid 2x1
    
    # With labels
    python stack_videos.py video1.mp4 video2.mp4 -o output.mp4 --labels "v_A=0.01" "v_A=0.02"
"""

import argparse
import math
from pathlib import Path

from moviepy import VideoFileClip, TextClip, CompositeVideoClip, clips_array


def parse_grid(grid_str):
    """Parse grid specification like '2x2' into (rows, cols)."""
    parts = grid_str.lower().split('x')
    if len(parts) != 2:
        raise ValueError(f"Invalid grid format: {grid_str}. Use format like '2x2' or '1x2'")
    return int(parts[0]), int(parts[1])


def add_label_to_clip(clip, label, font_size=24):
    """Add a text label to the top-left of a video clip."""
    if not label:
        return clip
    
    # Try several fonts that should be available on Windows
    fonts_to_try = [
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/calibri.ttf', 
        'C:/Windows/Fonts/segoeui.ttf',
        'C:/Windows/Fonts/consola.ttf',
        None  # Let moviepy use default
    ]
    
    for font in fonts_to_try:
        try:
            txt = TextClip(
                text=label,
                font_size=font_size,
                color='white',
                stroke_color='black',
                stroke_width=2,
                font=font
            )
            txt = txt.with_duration(clip.duration)
            txt = txt.with_position((10, 10))  # Top-left corner
            
            return CompositeVideoClip([clip, txt])
        except Exception as e:
            continue
    
    print(f"Warning: Could not add label '{label}' - no suitable font found")
    return clip


def stack_videos(video_paths, output_path, rows=1, cols=2, labels=None, 
                 fps=None, label_size=24):
    """Stack videos into a grid using moviepy."""
    
    # Check all input files exist
    for vp in video_paths:
        if not Path(vp).exists():
            raise FileNotFoundError(f"Video not found: {vp}")
    
    n_videos = len(video_paths)
    
    # Auto-determine grid if not fully specified
    if rows * cols < n_videos:
        cols = math.ceil(math.sqrt(n_videos))
        rows = math.ceil(n_videos / cols)
        print(f"Auto-adjusted grid to {rows}x{cols} to fit {n_videos} videos")
    
    # Load all video clips
    print("Loading videos...")
    clips = []
    for i, vp in enumerate(video_paths):
        print(f"  Loading {vp}")
        clip = VideoFileClip(str(vp))
        
        # Add label if provided
        if labels and i < len(labels):
            clip = add_label_to_clip(clip, labels[i], label_size)
        
        clips.append(clip)
    
    # Find the minimum duration (in case videos have different lengths)
    min_duration = min(c.duration for c in clips)
    clips = [c.subclipped(0, min_duration) for c in clips]
    
    # Build grid array
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            idx = r * cols + c
            if idx < len(clips):
                row.append(clips[idx])
            else:
                # Create black clip for empty cells
                from moviepy import ColorClip
                black = ColorClip(size=clips[0].size, color=(0, 0, 0), duration=min_duration)
                row.append(black)
        grid.append(row)
    
    # Combine into grid
    print("Combining videos...")
    final = clips_array(grid)
    
    # Set fps
    if fps:
        final = final.with_fps(fps)
    
    # Write output
    print(f"Writing {output_path}...")
    final.write_videofile(
        str(output_path),
        codec='libx264',
        audio=False,
        preset='medium',
        threads=4
    )
    
    # Close all clips
    for clip in clips:
        clip.close()
    final.close()
    
    print(f"\nSaved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Stack multiple videos into a grid layout',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Side by side:
    python stack_videos.py video1.mp4 video2.mp4 -o combined.mp4
    
  2x2 grid:
    python stack_videos.py v1.mp4 v2.mp4 v3.mp4 v4.mp4 -o grid.mp4 --grid 2x2
    
  Vertical stack:
    python stack_videos.py top.mp4 bottom.mp4 -o stacked.mp4 --grid 2x1
    
  With labels:
    python stack_videos.py low.mp4 high.mp4 -o compare.mp4 --labels "v_A=0.01" "v_A=0.05"
"""
    )
    
    parser.add_argument('videos', nargs='+', help='Input video files')
    parser.add_argument('-o', '--output', required=True, help='Output video file')
    parser.add_argument('--grid', default='1x2', 
                        help='Grid layout as ROWSxCOLS (default: 1x2 for side-by-side)')
    parser.add_argument('--labels', nargs='+', help='Labels for each video')
    parser.add_argument('--label-size', type=int, default=24, help='Font size for labels')
    parser.add_argument('--fps', type=int, help='Output frame rate (default: same as input)')
    
    args = parser.parse_args()
    
    rows, cols = parse_grid(args.grid)
    
    print(f"Stacking {len(args.videos)} videos in {rows}x{cols} grid")
    if args.labels:
        print(f"Labels: {args.labels}")
    
    stack_videos(
        args.videos,
        args.output,
        rows=rows,
        cols=cols,
        labels=args.labels,
        fps=args.fps,
        label_size=args.label_size
    )


if __name__ == '__main__':
    main()
