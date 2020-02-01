import argparse
import os

import uvloop
from aiohttp import web

EXPECTED_MODES = ['client', 'server']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str,
                        help='program mode client or server')
    args = parser.parse_args()
    mode = args.mode
    if mode not in EXPECTED_MODES:
        message = 'Invalid mode provided {} one of {}'.format(mode, EXPECTED_MODES)
        raise Exception(message)
    if mode == 'client':
        from video_service import VideoService
        RTSP_STREAM = os.environ['RTSP_STREAM']
        NTH_FRAME = int(os.environ['NTH_FRAME'])
        MARKET_CODE = os.environ['MARKET_CODE']
        SERVER_ADDRESS = os.environ['SERVER_ADDRESS']
        path = RTSP_STREAM
        while True:
            video_service = VideoService(
                path,
                "tracker_config.json",
                SERVER_ADDRESS,
                nth_frame=NTH_FRAME,
                market_code=MARKET_CODE
            )
            video_service.run()
    if mode == 'server':
        from finderserver import app
        uvloop.install()
        web.run_app(app)


