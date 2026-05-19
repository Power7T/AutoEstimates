"""Tool definitions for Jarvis's Claude AI brain."""

TOOLS = [
    {
        "name": "open_project",
        "description": "Open an existing CapCut project by name, or create a new one.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Name of the project to open or create.",
                },
                "create_new": {
                    "type": "boolean",
                    "description": "Set true to create a new project with this name.",
                    "default": False,
                },
            },
            "required": ["project_name"],
        },
    },
    {
        "name": "import_media",
        "description": "Import a video or image file into the current CapCut project.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the media file to import.",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "trim_clip",
        "description": "Trim a clip on the timeline to a specific start and end time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "clip_index": {
                    "type": "integer",
                    "description": "Zero-based index of the clip on the timeline.",
                    "default": 0,
                },
                "start_seconds": {
                    "type": "number",
                    "description": "New start time in seconds.",
                },
                "end_seconds": {
                    "type": "number",
                    "description": "New end time in seconds.",
                },
            },
            "required": ["start_seconds", "end_seconds"],
        },
    },
    {
        "name": "add_text",
        "description": "Add a text overlay to the video at a specific position and time range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text content to display.",
                },
                "position": {
                    "type": "string",
                    "enum": ["top", "center", "bottom"],
                    "description": "Vertical position of the text.",
                    "default": "bottom",
                },
                "start_seconds": {
                    "type": "number",
                    "description": "When the text appears (seconds from video start).",
                    "default": 0,
                },
                "duration_seconds": {
                    "type": "number",
                    "description": "How long the text stays on screen.",
                    "default": 3,
                },
                "font_size": {
                    "type": "string",
                    "enum": ["small", "medium", "large"],
                    "default": "medium",
                },
                "color": {
                    "type": "string",
                    "description": "Text color (e.g. 'white', 'yellow', '#FF0000').",
                    "default": "white",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "add_transition",
        "description": "Add a transition effect between two clips.",
        "input_schema": {
            "type": "object",
            "properties": {
                "clip_index": {
                    "type": "integer",
                    "description": "Index of the clip BEFORE the transition.",
                    "default": 0,
                },
                "transition_type": {
                    "type": "string",
                    "enum": [
                        "fade", "dissolve", "wipe_left", "wipe_right",
                        "zoom_in", "zoom_out", "flash", "glitch", "none",
                    ],
                    "description": "Type of transition to apply.",
                    "default": "fade",
                },
                "duration_seconds": {
                    "type": "number",
                    "description": "Duration of the transition in seconds.",
                    "default": 0.5,
                },
            },
            "required": ["transition_type"],
        },
    },
    {
        "name": "add_music",
        "description": "Add background music to the video from CapCut's music library or a local file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "enum": ["library", "local"],
                    "description": "'library' to search CapCut's built-in music, 'local' for a file on disk.",
                    "default": "library",
                },
                "query_or_path": {
                    "type": "string",
                    "description": "Search query for library music, or file path for local music.",
                },
                "volume": {
                    "type": "number",
                    "description": "Volume level 0.0–1.0.",
                    "default": 0.5,
                },
                "fade_in": {
                    "type": "boolean",
                    "description": "Apply fade-in at the start.",
                    "default": True,
                },
                "fade_out": {
                    "type": "boolean",
                    "description": "Apply fade-out at the end.",
                    "default": True,
                },
            },
            "required": ["query_or_path"],
        },
    },
    {
        "name": "apply_filter",
        "description": "Apply a color filter or visual effect to the entire video or a specific clip.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filter_name": {
                    "type": "string",
                    "description": "Name of the filter (e.g. 'vintage', 'cinematic', 'bright', 'moody', 'black_white', 'warm', 'cool').",
                },
                "intensity": {
                    "type": "number",
                    "description": "Filter strength from 0.0 to 1.0.",
                    "default": 0.7,
                },
                "apply_to": {
                    "type": "string",
                    "enum": ["all", "clip"],
                    "description": "Apply to the whole video or just one clip.",
                    "default": "all",
                },
                "clip_index": {
                    "type": "integer",
                    "description": "Which clip to apply to (if apply_to is 'clip').",
                    "default": 0,
                },
            },
            "required": ["filter_name"],
        },
    },
    {
        "name": "adjust_speed",
        "description": "Change the playback speed of a clip (slow motion or fast forward).",
        "input_schema": {
            "type": "object",
            "properties": {
                "clip_index": {
                    "type": "integer",
                    "description": "Index of the clip to adjust.",
                    "default": 0,
                },
                "speed_multiplier": {
                    "type": "number",
                    "description": "Speed multiplier: <1 for slow motion, >1 for fast forward (e.g. 0.5 = half speed, 2.0 = double speed).",
                },
            },
            "required": ["speed_multiplier"],
        },
    },
    {
        "name": "adjust_volume",
        "description": "Adjust the audio volume of a clip.",
        "input_schema": {
            "type": "object",
            "properties": {
                "clip_index": {
                    "type": "integer",
                    "description": "Index of the clip.",
                    "default": 0,
                },
                "volume": {
                    "type": "number",
                    "description": "Volume from 0.0 (mute) to 1.0 (full) or higher for boost.",
                },
            },
            "required": ["volume"],
        },
    },
    {
        "name": "add_sticker",
        "description": "Add an animated sticker or emoji overlay to the video.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term for the sticker (e.g. 'fire', 'heart', 'star', 'laugh').",
                },
                "position_x": {
                    "type": "number",
                    "description": "Horizontal position as percentage of screen width (0–100).",
                    "default": 50,
                },
                "position_y": {
                    "type": "number",
                    "description": "Vertical position as percentage of screen height (0–100).",
                    "default": 50,
                },
                "start_seconds": {
                    "type": "number",
                    "description": "When the sticker appears.",
                    "default": 0,
                },
                "duration_seconds": {
                    "type": "number",
                    "description": "How long the sticker shows.",
                    "default": 2,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "crop_video",
        "description": "Crop or reframe the video to a different aspect ratio.",
        "input_schema": {
            "type": "object",
            "properties": {
                "aspect_ratio": {
                    "type": "string",
                    "enum": ["9:16", "16:9", "1:1", "4:3", "3:4", "original"],
                    "description": "Target aspect ratio.",
                },
            },
            "required": ["aspect_ratio"],
        },
    },
    {
        "name": "export_video",
        "description": "Export/render the final video with chosen quality settings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "resolution": {
                    "type": "string",
                    "enum": ["480p", "720p", "1080p", "2k", "4k"],
                    "description": "Export resolution.",
                    "default": "1080p",
                },
                "fps": {
                    "type": "integer",
                    "enum": [24, 30, 60],
                    "description": "Frames per second.",
                    "default": 30,
                },
                "format": {
                    "type": "string",
                    "enum": ["mp4", "mov"],
                    "description": "Output file format.",
                    "default": "mp4",
                },
            },
            "required": [],
        },
    },
    {
        "name": "undo",
        "description": "Undo the last action in CapCut.",
        "input_schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "integer",
                    "description": "Number of actions to undo.",
                    "default": 1,
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_timeline_info",
        "description": "Get information about the current timeline: clips, duration, and current state.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]
