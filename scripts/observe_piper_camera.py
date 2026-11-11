import socket,struct,select,time,json,csv,argparse
from pathlib import Path
import numpy as np
import cv2
import pyrealsense2 as rs
parser=argparse.ArgumentParser(description='Receive-only CAN and central-camera capture; never sends CAN frames.')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--seconds',type=float,default=180)
parser.add_argument('--allow-enabled',action='store_true',help='Observe a separately controlled arm; never use this mode for manual handling')
parser.add_argument('--raw-can',action='store_true',help='Record received CAN payloads and local-loopback flags for command/feedback comparison')
args=parser.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=False)
buses={};frames={};history={};rows=0
for name in ('can_arm1','can_arm2'):
 s=socket.socket(socket.AF_CAN,socket.SOCK_RAW,socket.CAN_RAW);s.bind((name,));s.setblocking(False);buses[name]=s;frames[name]={};history[name]=[]
ctx=rs.context();dev=next(d for d in ctx.query_devices() if '455' in d.get_info(rs.camera_info.name));serial=dev.get_info(rs.camera_info.serial_number)
p=rs.pipeline();c=rs.config();c.enable_device(serial);c.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
f=(out/'observations.csv').open('w',newline='');w=csv.writer(f);w.writerow(['time_monotonic','frame','can','enabled_count']+[f'j{i}_deg' for i in range(1,7)])
camera_log=(out/'camera_frames.csv').open('w',newline='');camera_writer=csv.writer(camera_log)
camera_writer.writerow(['time_monotonic','video_frame','device_frame_number','device_timestamp_ms','timestamp_domain'])
raw_log=(out/'raw_can.jsonl').open('w') if args.raw_can else None
v=cv2.VideoWriter(str(out/'scene.avi'),cv2.VideoWriter_fourcc(*'MJPG'),30,(640,480))
assert v.isOpened()
start=time.monotonic();last_image=-1
try:
 p.start(c)
 print('RECEIVE_ONLY_RECORDING_STARTED: no CAN sends',flush=True)
 while time.monotonic()-start < args.seconds:
  frame=p.wait_for_frames(3000);color=frame.get_color_frame();img=np.asanyarray(color.get_data());now=time.monotonic();v.write(img)
  camera_writer.writerow([now,rows,color.get_frame_number(),color.get_timestamp(),str(color.get_frame_timestamp_domain())])
  for name,s in buses.items():
   for _ in range(3000):
    try: raw,ancillary,flags,address=s.recvmsg(16)
    except BlockingIOError: break
    ident,dlc,data=struct.unpack('=IB3x8s',raw)
    if raw_log:
     raw_log.write(json.dumps({'received_monotonic':time.monotonic(),'video_frame':rows,'can':name,'id':ident,'dlc':dlc,'data_hex':data[:dlc].hex(),'host_local':bool(flags&socket.MSG_DONTROUTE)})+'\n')
    if dlc==8:frames[name][ident]=(now,data)
   required={0x2a5,0x2a6,0x2a7,*range(0x261,0x267)}
   if all(i in frames[name] and now-frames[name][i][0]<.3 for i in required):
    q=[a/1000 for i in (0x2a5,0x2a6,0x2a7) for a in struct.unpack('>ii',frames[name][i][1])]
    enabled=sum(bool(frames[name][i][1][5]&64) for i in range(0x261,0x267))
    w.writerow([now,rows,name,enabled,*q]);history[name].append(q)
    if enabled and not args.allow_enabled:raise RuntimeError('Unexpected enabled motors on '+name+'; stop manual handling')
  f.flush();camera_log.flush()
  if raw_log:raw_log.flush()
  rows+=1
  second=int(now-start)
  if second!=last_image:
   cv2.imwrite(str(out/f'frame_{second:03d}.jpg'),img);last_image=second
  if (out/'STOP').exists():break
finally:
 p.stop();v.release();f.close();camera_log.close()
 if raw_log:raw_log.close()
 for s in buses.values():s.close()
 report={'mode':'receive-only; zero CAN sends','camera_serial':serial,'frames':rows,'duration_s':time.monotonic()-start,'channels':{}}
 for name,h in history.items():
  if h:
   a=np.array(h);report['channels'][name]={'first_deg':h[0],'last_deg':h[-1],'range_deg':np.ptp(a,axis=0).tolist()}
 (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
