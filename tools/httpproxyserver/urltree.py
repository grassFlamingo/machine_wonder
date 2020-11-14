import re


class URLTree:
    """
    ref: rfc1035

    <letters> := a-zA-Z
    <digit> := 0-9
    <hyp> := -
    """

    class TNode:
        def __init__(self, name="*") -> None:
            self.nodes = []
            self.name = name

        def search(self, item):
            if len(self.nodes) > 0 and self.nodes[0] == "*":
                return self.nodes[0]
            if item is None:
                return None
            for n in self.nodes:
                if n == item:
                    return n
            return None

        def append(self, item: str):
            ans = self.search(item)
            if ans != None:
                return ans
            ans = URLTree.TNode(item)
            if item == "*":
                # just rewrite anything with *
                self.nodes.clear()
            self.nodes.append(ans)
            return ans

        def has_sublings(self):
            return len(self.nodes) > 0

        def __str__(self, index=1) -> str:
            ans = self.name + "\n"
            if len(self.nodes) == 0:
                return ans

            for n in self.nodes:
                ans += " "*index + n.__str__(index+1)
            return ans

        def __eq__(self, other):
            if isinstance(other, str):
                return other == self.name
            elif isinstance(other, self.__class__):
                return other.name == self.name
            else:
                return False

    def __init__(self) -> None:
        self.urlnodes = URLTree.TNode("@ROOT")
        self.reg = re.compile("([*a-zA-Z0-9\\-]+\\.)+[a-zA-Z0-9]")

    def __str__(self) -> str:
        return f"URLTree:\n{self.urlnodes}"

    def push_url(self, hostname: str) -> None:
        if not self.reg.match(hostname):
            # not a host name
            print(f"[not a hostname] {hostname}")
            return

        subdomains = hostname.split(".")
        tn = self.urlnodes

        # in reverse order
        for sd in reversed(subdomains):
            tn = tn.append(sd)

    def __contains__(self, hostname):
        if not self.reg.match(hostname):
            # not a host name
            print(f"[not a hostname] {hostname}")
            return
        subdomains = hostname.split(".")
        tn = self.urlnodes
        for sd in reversed(subdomains):
            _tn = tn.search(sd)
            if _tn is None:
                return False
            if _tn == "*":
                return True
            tn = _tn

        return not tn.has_sublings()
