r"""

# HTTPProxy Server with block list

1. Connect your server and client  to same LAN.
2. set http proxy for clint as <server>:<port>
3. run server with "python httpproxyserver.py --port <port> --blist httpproxyserver_blist.txt"
4. run -------


## about block list

- the line start by '#' will be ignored
- empty line will also be ignored
- * means wild cast (only * in the begining is supported now)
- each domain name is end by \n

"""
import asyncio
import os
import urllib.parse
import logging
import argparse
from urltree import URLTree

parser = argparse.ArgumentParser(description='Http Proxy Server')
parser.add_argument("--bind", "-b", type=str, default="0.0.0.0",
                    help="address to bind, default is 0.0.0.0")
parser.add_argument("--port", "-p", type=int, default=8080,
                    help='ports to listen, default is 8080')
parser.add_argument("--blist", "-l", type=str, default="httpproxyserver_blist.txt",
                    help="the block list for httpproxyserver; separate by \\n")

args = parser.parse_args()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(filename)s %(levelname)s: %(message)s')


GBlockTree = URLTree()  # global block tree


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
               **kwargs):
    try:
        while not reader.at_eof():
            writer.write(await reader.read(1024))
    except Exception as e:
        logging.error(e)


# ref: https://tools.ietf.org/html/draft-luotonen-web-proxy-tunneling-01
async def handle_ssl(client_reader: asyncio.StreamReader,
                     client_writer: asyncio.StreamWriter, hostport,
                     **kwargs):
    logging.info(f"[start ssl] -> {hostport}")
    try:
        remote_reader, remote_writer = await asyncio.open_connection(*hostport)
    except Exception as e:
        logging.error(e)
        logging.info(f"[end ssl] <- {hostport}")
        return

    resp = f"{kwargs['httpversion']} 200 Connection established\r\nProxy-agent: Netscape-Proxy/1.1\r\n\r\n"
    client_writer.write(resp.encode())

    try:
        up = pipe(remote_reader, client_writer)
        down = pipe(client_reader, remote_writer)
        await asyncio.gather(up, down)
    except Exception as e:
        logging.error(e)

    remote_writer.close()
    logging.info(f"[end ssl] <- {hostport}")


async def handle_http(client_reader: asyncio.StreamReader,
                      client_writer: asyncio.StreamWriter, hostport,
                      **kwargs):
    logging.info(f"[start http] -> {hostport}")
    try:
        remote_reader, remote_writer = await asyncio.open_connection(*hostport)
    except Exception as e:
        logging.error(e)
        logging.info(f"[end http] <- {hostport}")
        return

    remote_writer.write(kwargs['headraw'])

    try:
        up = pipe(remote_reader, client_writer)
        down = pipe(client_reader, remote_writer)
        await asyncio.gather(up, down)
    except Exception as e:
        logging.error(e)

    remote_writer.close()
    logging.info(f"[end http] <- {hostport}")


async def handle_queries(client_reader: asyncio.StreamReader,
                         client_writer: asyncio.StreamWriter):
    header_raw = await client_reader.read(512)
    if not header_raw:
        return
    header = header_raw.decode().split("\r\n")
    hellomsg = header[0].split()

    headerdict = dict()
    for l in header[1:-1]:
        ti = l.find(":")
        headerdict[l[0:ti]] = l[ti + 1::]
    headerdict['headraw'] = header_raw

    if hellomsg[0] == "CONNECT":
        # https connection
        thp = hellomsg[1].split(":")
        outhost = thp[0]
        if outhost in GBlockTree:
            logging.info(f"[blocked] {outhost}")
            client_writer.close()
            return
        outport = int(thp[1])
        headerdict["httpversion"] = hellomsg[2]
        await handle_ssl(client_reader, client_writer,
                         (outhost, outport), **headerdict)
    elif hellomsg[0] == "GET" or hellomsg[0] == "POST":
        # http connection
        # GET http://www.baidu.com/ HTTP/1.1
        parans = urllib.parse.urlparse(hellomsg[1])
        outhost = parans.hostname

        if outhost in GBlockTree:
            logging.info(f"[blocked] {outhost}")
            client_writer.close()
            return

        outport = 80 if parans.port is None else parans.port
        await handle_http(client_reader, client_writer,
                          (outhost, outport), **headerdict)

    client_writer.close()

# read block list
if os.path.exists(args.blist):
    with open(args.blist) as bfile:
        for l in bfile:
            l = l.strip()
            # comment line
            if len(l) <= 1 or l[0] == '#':
                continue
            GBlockTree.push_url(l)


loop = asyncio.get_event_loop()
server = loop.run_until_complete(
    asyncio.start_server(handle_queries, args.bind, args.port))
logging.info(f"[system] running http proxy on http://{args.bind}:{args.port}")

try:
    loop.run_forever()
except Exception as e:
    logging.error(e)
    logging.info("[system] exit")
    server.close()
    loop.close()
