# Deployment

Status: Not deployed yet (template).

- Planned URL: `https://neojustin.dothost.net/p/2d-animation-lora-pipeline/`
- Planned docker-compose service name: `2d-animation-lora-pipeline`
- Planned server checkout path: `/home/neojustin/justin-portfolio/projects/2d-animation-lora-pipeline`

## Deploy (when dockerized)
1) Add Docker config to this repo (committed).
2) Add a service to `/home/neojustin/justin-portfolio/docker-compose.yml` with name `2d-animation-lora-pipeline`.
3) Build + start on the server:
```bash
cd /home/neojustin/justin-portfolio
docker-compose up -d --build 2d-animation-lora-pipeline
```

## Update after code changes (once deployed)
```bash
cd /home/neojustin/justin-portfolio
docker-compose up -d --build 2d-animation-lora-pipeline
```

Reference workflow:
- `/home/justin/web-projects/justin-portfolio/docs/deployment/update-workflow.md`

