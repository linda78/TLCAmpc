#cloud-config

package_update: true
package_upgrade: true

packages:
  - python3
  - python3-pip
  - python3-venv
  - git
  - build-essential
  - libpq-dev

runcmd:
  - mkdir -p /root/.ssh
  - echo "${deploy_key_b64}" | base64 -d > /root/.ssh/deploy_key
  - chmod 600 /root/.ssh/deploy_key
  - printf "Host github.com\n  IdentityFile /root/.ssh/deploy_key\n  StrictHostKeyChecking no\n" > /root/.ssh/config
  - chmod 600 /root/.ssh/config
  - GIT_SSH_COMMAND="ssh -i /root/.ssh/deploy_key -o StrictHostKeyChecking=no" git clone -b ${git_branch} ${git_repo_url} /opt/TLCAmpc
  - python3 -m venv /opt/TLCAmpc/.venv
  - /opt/TLCAmpc/.venv/bin/pip install --upgrade pip
  - /opt/TLCAmpc/.venv/bin/pip install -e /opt/TLCAmpc
  - /opt/TLCAmpc/.venv/bin/pip install psycopg2-binary
  - /opt/TLCAmpc/.venv/bin/pip uninstall -y PySide6 PySide6-Essentials PySide6-Addons shiboken6 || true
  - printf "[Unit]\nDescription=Drone Simulation Worker\nAfter=network-online.target\nWants=network-online.target\n\n[Service]\nType=simple\nUser=root\nWorkingDirectory=/opt/TLCAmpc\nEnvironment=DRONESIM_DATABASE_URL=postgresql://dronesim:${db_password}@${db_private_ip}:5432/dronesim\nEnvironment=PYTHONPATH=/opt/TLCAmpc:/opt/TLCAmpc/src\nEnvironment=OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\nExecStart=/opt/TLCAmpc/.venv/bin/python -m paper2_tools.distributed.worker --n-workers -1 --v-max ${v_max} --u-max ${u_max} --room-size ${room_size} --static-safety-zone ${static_safety_zone} --adaptive-safety-zone ${adaptive_safety_zone} --r-min ${r_min} --log-level INFO\nRestart=on-failure\nRestartSec=30\n\n[Install]\nWantedBy=multi-user.target\n" > /etc/systemd/system/sim-worker.service
  - systemctl daemon-reload
  - systemctl enable sim-worker
  - systemctl start sim-worker
