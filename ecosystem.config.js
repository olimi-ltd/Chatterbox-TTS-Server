module.exports = {
  apps: [
    {
      name: "chatterbox-tts",
      script: ".venv/bin/python",
      args: "server.py",
      cwd: "/home/unknown/projects/incode/chatterbox/Chatterbox-TTS-Server",
      interpreter: "none",
      env: {
        PYTHONUNBUFFERED: "1",
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "4G",
      log_date_format: "YYYY-MM-DD HH:mm:ss Z",
      error_file: "logs/pm2-error.log",
      out_file: "logs/pm2-out.log",
      merge_logs: true,
      min_uptime: "30s",
      max_restarts: 5,
      restart_delay: 5000,
    },
  ],
};
