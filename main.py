# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio
import base64
import io
import os
import sys
import traceback

import pyaudio
import argparse
from dotenv import load_dotenv

# Conditional imports for video modes
cv2 = None
mss = None
PILImage = None

from google import genai
from google.genai import types

from mcp_handler import MCPClient

load_dotenv()

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-exp"

DEFAULT_MODE = "camera"

client = genai.Client(http_options={"api_version": "v1alpha"})

# While Gemini 2.0 Flash is in experimental preview mode, only one of AUDIO or
# TEXT may be passed here.
# CONFIG = {"tools": tools, "response_modalities": ["AUDIO"]}

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE, text_only=False):
        self.video_mode = video_mode
        self.text_only = text_only

        # Import video dependencies only if needed
        if video_mode in ("camera", "screen"):
            global cv2, mss, PILImage
            import cv2
            import mss
            from PIL import Image as PILImage

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None
        self.audio_stream = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

        self.mcp_client = MCPClient()

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)
            
    def handle_server_content(self, server_content):
        model_turn = server_content.model_turn
        if model_turn:
            for part in model_turn.parts:
                executable_code = part.executable_code
                if executable_code is not None:
                    print('-------------------------------')
                    print(f'``` python\n{executable_code.code}\n```')
                    print('-------------------------------')

                code_execution_result = part.code_execution_result
                if code_execution_result is not None:
                    print('-------------------------------')
                    print(f'```\n{code_execution_result.output}\n```')
                    print('-------------------------------')

        grounding_metadata = getattr(server_content, 'grounding_metadata', None)
        if grounding_metadata is not None:
            print(grounding_metadata.search_entry_point.rendered_content)

        return
    
    async def handle_tool_call(self, tool_call):
        for fc in tool_call.function_calls:
            result = await self.mcp_client.call_tool(fc.name, fc.args or {})
            print(result)
            tool_response = types.LiveClientToolResponse(
                function_responses=[types.FunctionResponse(
                    name=fc.name,
                    id=fc.id,
                    response={'result': result},
                )]
            )

            print('\n>>> ', tool_response)
            await self.session.send(input=tool_response)

    def _get_frame(self, cap):
        # Read the frameq
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PILImage.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

        # Release the VideoCapture object
        cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PILImage.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        print('Microphone:', mic_info['name'])
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")
                    
                server_content = response.server_content
                if server_content is not None:
                    self.handle_server_content(server_content)
                    continue

                tool_call = response.tool_call
                if tool_call is not None:
                    await self.handle_tool_call(tool_call)

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def receive_text_only(self):
        """Receive text responses only (no audio)."""
        while True:
            turn = self.session.receive()
            async for response in turn:
                if text := response.text:
                    print(text, end="", flush=True)

                server_content = response.server_content
                if server_content is not None:
                    self.handle_server_content(server_content)
                    continue

                tool_call = response.tool_call
                if tool_call is not None:
                    await self.handle_tool_call(tool_call)
            print()  # newline after turn complete

    async def run(self):
        await self.mcp_client.connect_to_server()
        all_tools = await self.mcp_client.list_all_tools()

        functional_tools = []
        for server_name, tool in all_tools:
            tool_desc = {
                "name": tool.name,
                "description": tool.description
            }
            if tool.inputSchema.get("properties"):
                tool_desc["parameters"] = {
                    "type": tool.inputSchema["type"],
                    "properties": {},
                }
                for param in tool.inputSchema["properties"]:
                    tool_desc["parameters"]["properties"][param] = {
                        "type": tool.inputSchema["properties"][param].get("type", "string"),
                        "description": tool.inputSchema["properties"][param].get("description", ""),
                    }

            if "required" in tool.inputSchema:
                tool_desc["parameters"]["required"] = tool.inputSchema["required"]

            functional_tools.append(tool_desc)
        print(f"Loaded {len(functional_tools)} tools from MCP servers")
        tools = [
            {
                'function_declarations': functional_tools,
                'code_execution': {},
                'google_search': {},
                },
        ]
        
        if self.text_only:
            CONFIG = {"tools": tools, "response_modalities": ["TEXT"]}
            print("Running in TEXT-ONLY mode (no audio)")
        else:
            CONFIG = {"tools": tools, "response_modalities": ["AUDIO"]}

        max_retries = 5
        retry_count = 0

        while retry_count < max_retries:
            try:
                async with (
                    client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session = session
                    retry_count = 0  # Reset on successful connection

                    self.audio_in_queue = asyncio.Queue()
                    self.out_queue = asyncio.Queue(maxsize=5)

                    send_text_task = tg.create_task(self.send_text())

                    if not self.text_only:
                        tg.create_task(self.send_realtime())
                        tg.create_task(self.listen_audio())
                        tg.create_task(self.receive_audio())
                        tg.create_task(self.play_audio())
                    else:
                        tg.create_task(self.receive_text_only())

                    if self.video_mode == "camera":
                        tg.create_task(self.get_frames())
                    elif self.video_mode == "screen":
                        tg.create_task(self.get_screen())

                    await send_text_task
                    raise asyncio.CancelledError("User requested exit")

            except asyncio.CancelledError:
                break
            except (ExceptionGroup, Exception) as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"\n[Connection error, retrying {retry_count}/{max_retries}...]")
                    await asyncio.sleep(2)
                else:
                    print(f"\n[Max retries reached, exiting]")
                    if self.audio_stream:
                        self.audio_stream.close()
                    traceback.print_exception(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="text input only, no audio",
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode, text_only=args.text_only)
    asyncio.run(main.run())