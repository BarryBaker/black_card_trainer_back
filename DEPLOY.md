# Deploy the trainer API

The trainer runs as one Gunicorn container in a separate Compose project. On
the Linux VPS it uses host networking to reach `blackcard_trainer` as
`blackcard_reader` at `127.0.0.1:5433`. Caddy reaches the trainer on
`127.0.0.1:8001`. These ports are loopback-only.

## Start the container on the VPS

```bash
git clone --branch main https://github.com/BarryBaker/black_card_trainer_back.git ~/blackCardTrainer
cd ~/blackCardTrainer
sudo ss -ltn 'sport = :8001'
test -e .env || install -m 600 .env.example .env
nano .env
```

Check that port `8001` has no listener before starting. Set `DB_PASSWORD` to
the password for `blackcard_reader`. Keep the single quotes so Compose treats
`$` literally, and escape any single quote in the password with `\'`. Set
`CORS_ALLOWED_ORIGINS` to the exact frontend origin, such as
`https://your-site.netlify.app`, with no trailing slash. Separate multiple
origins with commas.

```bash
sudo docker compose up -d --build
sudo docker compose ps
curl -fsS http://127.0.0.1:8001/api/health
curl -fsS http://127.0.0.1:8001/api/db/health
```

The second health check should report `blackcard_trainer`, `blackcard_reader`,
and `"query_result":1`. Use `sudo docker compose logs --tail=50 api` if a
health check fails.

## Add the HTTPS route

On the VPS, add the following site to the existing Caddyfile. Substitute the
VPS address with dashes for `YOUR-VPS-IP-WITH-DASHES`. The public host serves
the frontend routes and app health check. The detailed database health check
is available through the local port.

```caddyfile
trainer.YOUR-VPS-IP-WITH-DASHES.sslip.io {
    @trainer_api path /api/health /api/db/lines /api/db/train /api/db/tree
    handle @trainer_api {
        reverse_proxy 127.0.0.1:8001
    }
    handle {
        respond 404
    }
}
```

For the systemd Caddy service with the standard config path, validate and
reload with these commands:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
sudo systemctl reload caddy
```

## Point Netlify at the API

Set the Netlify build variable `VITE_BASE_URL` to
`https://trainer.YOUR-VPS-IP-WITH-DASHES.sslip.io` and redeploy the frontend.
The frontend appends `/api/db/lines`, `/api/db/train`, and `/api/db/tree`.
To verify TLS and CORS, send a preflight request using the deployed site origin:

```bash
curl -fsSi -X OPTIONS 'https://trainer.YOUR-VPS-IP-WITH-DASHES.sslip.io/api/db/train' \
  -H 'Origin: https://your-site.netlify.app' \
  -H 'Access-Control-Request-Method: POST' \
  -H 'Access-Control-Request-Headers: Content-Type'
```

Look for `Access-Control-Allow-Origin` with the exact Netlify origin.

## Deploy an update

```bash
cd ~/blackCardTrainer
git pull --ff-only origin main
sudo docker compose up -d --build
```
