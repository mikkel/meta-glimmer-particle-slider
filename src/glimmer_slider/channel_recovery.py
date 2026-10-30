"""Conservative display recovery for explicitly addressed malformed model output.

Missing protocol separators remain reported as malformed. Unaddressed text is
never relabeled as reasoning; the original generated stream remains available.
"""
import re

START = '<|start|>assistant'
MESSAGE = '<|message|>'
CLOSE = ('<|eom|>','<|eot|>','<|end_of_text|>')
HEADER = re.compile(r'<\|start\|>assistant[ \t\r\n]+to=(self|user)(?=$|\s|<|\.\.\.)')
MARKERS = (START, START+' to=self'+MESSAGE, START+' to=user'+MESSAGE, MESSAGE, *CLOSE)


def _without_partial_marker(text):
    for pos in range(max(0,len(text)-48),len(text)):
        tail=text[pos:]
        if any(marker.startswith(tail) for marker in MARKERS):return text[:pos]
    return text


def recover_explicit_channels(raw, prefix):
    # Only the unfinished assistant suffix is context. Never scan prompt text.
    at=prefix.rfind(START)
    pending=prefix[at:] if at>=0 else ''
    if pending not in (START,START+' to=user'+MESSAGE,START+' to=self'+MESSAGE):
        pending=''
    combined=pending+raw
    headers=list(HEADER.finditer(combined))
    parts={'self':[],'user':[]};warnings=[];recognized=False;pending_header=False
    for i,head in enumerate(headers):
        body=combined[head.end():headers[i+1].start() if i+1<len(headers) else len(combined)]
        # An incomplete real delimiter is pending, not a missing delimiter.
        if not body or (body.startswith('<') and MESSAGE.startswith(body)):
            pending_header=True;continue
        valid=body.startswith(MESSAGE)
        if valid:body=body[len(MESSAGE):]
        else:
            stripped=body.lstrip()
            if stripped.startswith('<') and MESSAGE.startswith(stripped):
                pending_header=True;continue
            if stripped.startswith(MESSAGE):body=stripped[len(MESSAGE):]
            warnings.append('Missing or displaced <|message|> after explicit to='+head.group(1))
        recognized=True
        stops=[body.find(marker) for marker in CLOSE if marker in body]
        if stops:body=body[:min(stops)]
        # Never leak a partial next assistant/header marker into displayed text.
        body=_without_partial_marker(body)
        parts[head.group(1)].append(body)
    reasoning=''.join(parts['self']);content=''.join(parts['user'])
    plain=re.sub(r'<\|[^>]+\|>','',raw).strip()
    if warnings:status='recovered_explicit_channels'
    elif recognized:status='standard'
    elif pending_header:status='pending_channel_header'
    elif plain and not any(value.startswith(plain) for value in ('to=self','to=user','<|start|>assistant')):status='unclassified'
    elif raw and not plain:status='special_tokens_only'
    else:status='pending_channel_header'
    return dict(reasoning=reasoning,content=content,status=status,warnings=list(dict.fromkeys(warnings)))
