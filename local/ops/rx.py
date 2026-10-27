"""run a remote command: python rx.py "cmd"
fetch:  python rx.py --get remote local
upload: python rx.py --put local remote"""
import os  # public release: SSH configuration comes from the environment
import sys, warnings
warnings.filterwarnings("ignore")
import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(os.environ["SSH_HOST"], port=int(os.environ.get("SSH_PORT", "22")), username=os.environ["SSH_USER"], password=os.environ.get("SSH_PASSWORD"), timeout=30)
if sys.argv[1] == "--get":
    s = c.open_sftp(); s.get(sys.argv[2], sys.argv[3])
elif sys.argv[1] == "--put":
    s = c.open_sftp(); s.put(sys.argv[2], sys.argv[3])
else:
    _, o, e = c.exec_command(sys.argv[1], timeout=1700)
    sys.stdout.buffer.write(o.read()); sys.stdout.buffer.write(e.read())
c.close()
