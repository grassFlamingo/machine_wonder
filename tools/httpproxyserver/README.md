# Http proxy server

juster a http proxy server, and we can block url from `blist`

## Usage

```
usage: httpproxyserver.py [-h] [--bind BIND] [--port PORT] [--blist BLIST]

Http Proxy Server

optional arguments:
  -h, --help            show this help message and exit
  --bind BIND, -b BIND  address to bind, default is 0.0.0.0
  --port PORT, -p PORT  ports to listen, default is 8080
  --blist BLIST, -l BLIST
                        the block list for httpproxyserver; separate by \n
```