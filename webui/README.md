# Open WebUI setup

Make sure docker is installed.

Make sure the Ollama server is up and running.
If not see [`ollama/README.md`](ollama/README.md).

Running both scripts in this directory should do the trick.

## Accessing the server

### On the same machine

Visit `http://localhost:8080` with your web browser.

### On another machine with an SSH tunnel

Open a shell, run the following command with your credentials.

```sh
ssh -i ~/.ssh/<key> -L 8080:localhost:8080 <user>@<ip>
```

Visit `http://localhost:8080` with your web browser.

### Make Open WebUI publically accessible

I'd strongly recommend considering the security of doing so first.
1. Control who can create users. Especially with admin access: Open WebUI
	allows uploading arbitrary Python scripts.
2. Consider that the current docker containers are using the host's network.
	Consider switching to isolating the docker container network.
3. Change the default ports and keys.

Have Open WebUI listen on 0.0.0.0 uncomment the following line in `webui.sh`:

```s
	-e HOST=0.0.0.0 \
```

Visit `http://<server_ip>:8080` with your web browser.
