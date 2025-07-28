#!/usr/bin/env python3

import asyncio
import json
from datetime import datetime

import websockets
import sys
import wave

async def run_test(uri):
    async with websockets.connect(uri) as websocket:
        start = datetime.now()
        wf = wave.open(sys.argv[1], "rb")
        await websocket.send('{ "config" : { "sample_rate" : %d } }' % (wf.getframerate()))
        buffer_size = int(wf.getframerate() * 0.2) # 0.2 seconds of audio
        results = []
        start2 = datetime.now()
        while True:

            data = wf.readframes(buffer_size)

            if len(data) == 0:
                break

            await websocket.send(data)
            res =await websocket.recv()
            result, text = print_text(res)
            if result:
                continue
            else:
                print_recognize_time(start2)
                start2 = datetime.now()
                results.append(text)

        await websocket.send('{"eof" : 1}')
        res = await websocket.recv()
        result, text = print_text(res)
        print_recognize_time(start2)
        results.append(text)
        all_text = ". ".join(results)
        end = datetime.now()
        elapsed = end - start
        print (f'recognize time {elapsed}')
        print(all_text)


def print_recognize_time(start):
    elapsed = datetime.now() - start
    print(f'recognize time {elapsed}')


def print_text(value):
    jres = json.loads(value)
    if not 'result' in jres:
        return True, ""
    print(jres['text'])
    return False, jres['text']

asyncio.run(run_test('ws://localhost:2700'))
