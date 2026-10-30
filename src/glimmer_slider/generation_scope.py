"""Per-prediction raw-adapter gating; this does not isolate cached hidden state."""
from contextlib import contextmanager
import re

START='<|start|>assistant'
HEADER=re.compile(r'^<\|start\|>assistant\s+to=(self|user)(?=$|\s|<|\.\.\.)')


def active_channel(prefix, raw):
    # Never inspect user-supplied body text: only the unfinished assistant suffix.
    pos=prefix.rfind(START)
    suffix=prefix[pos:] if pos>=0 else ''
    if suffix not in (START,START+' to=user<|message|>',START+' to=self<|message|>'):suffix=''
    text=suffix+raw
    at=text.rfind(START)
    if at<0:return 'unknown'
    tail=text[at:]
    match=HEADER.match(tail)
    if not match:return 'unknown'
    body=tail[match.end():]
    # Terminal or partially emitted delimiters await a new explicit recipient.
    if any(marker in body for marker in ('<|eom|>','<|eot|>','<|end_of_text|>')):return 'unknown'
    for marker in ('<|eom|>','<|eot|>','<|start|>assistant','<|end_of_text|>'):
        if any(tail.endswith(marker[:n]) for n in range(2,len(marker))):return 'unknown'
    return 'thinking' if match.group(1)=='self' else 'response'


class ScopeGate:
    def __init__(self, bank, strengths, scopes, snapshot, initial):
        self.bank=bank;self.channel=None;self.transitions=[]
        self.scopes=dict(scopes);self.enabled=any(v!='both' and strengths.get(k,0) for k,v in self.scopes.items())
        self.plans={}
        if self.enabled:
            for channel in ('thinking','response','unknown'):
                values={k:v for k,v in strengths.items() if self.scopes.get(k,'both') in ('both',channel)}
                self.plans[channel]=bank.compile(values,snapshot)
        self.initial=initial

    def select(self, channel):
        if channel==self.channel:return
        self.channel=channel;self.transitions.append(channel)
        if self.enabled:self.bank._active.set(self.plans[channel])

    @contextmanager
    def using(self):
        token=self.bank._active.set(self.bank._active.get()) if self.enabled else None
        try:
            self.select(self.initial);yield self
        finally:
            if token is not None:self.bank._active.reset(token)

    def report(self):
        return dict(scopes=self.scopes,transitions=self.transitions,unknown_policy='Both stays active; Thinking/Response are off while recipient is unknown',
            semantics='Gates next prediction at token callbacks, including prompt prefill. Boundary tokens use the preceding gate; KV state may carry prior influence. Not answer isolation.')
