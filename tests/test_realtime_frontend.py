"""Exercise the real inline controller with browser doubles, without a webcam.

Node is optional for Python-only installations. Real browser/media behavior is
checked separately; these tests cover sequencing, cleanup and stale callbacks.
"""

from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess

import pytest

FRONTEND = Path(__file__).resolve().parents[1] / "src/cv_human_video_matting/realtime/static/index.html"


class Inventory(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def test_page_is_self_contained_and_has_explicit_local_video_source():
    page = Inventory()
    page.feed(FRONTEND.read_text(encoding="utf-8"))
    assert not any(tag == "script" and attrs.get("src") for tag, attrs in page.elements)
    assert not any(tag == "link" and attrs.get("href") for tag, attrs in page.elements)
    assert any(tag == "input" and attrs.get("type") == "file" and attrs.get("accept") == "video/*"
               for tag, attrs in page.elements)
    assert not any(tag == "video" and (attrs.get("src") or "loop" in attrs) for tag, attrs in page.elements)


HARNESS = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const code = fs.readFileSync(process.argv[1], 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
function deferred() { let resolve, reject; const promise = new Promise((a,b) => {resolve=a; reject=b;}); return {promise,resolve,reject}; }
async function flush() { for (let i=0;i<8;i++) await Promise.resolve(); }
function browser(url='http://127.0.0.1:7990/') {
  const h = {now:0, cameraCalls:0, trackStops:0, playCalls:0, bitmapCloses:0, urls:[], revoked:[], sockets:[], timers:new Map(), rafs:new Map(), next:1};
  class Element {
    constructor(id) { this.id=id; this.dataset={}; this.textContent=''; this.value=''; this.listeners=new Map(); this.files=[]; this.width=640; this.height=360; this.draws=[]; this.readyState=4; this.videoWidth=1280; this.videoHeight=720; }
    addEventListener(name, fn) { if(!this.listeners.has(name)) this.listeners.set(name,new Set()); this.listeners.get(name).add(fn); }
    removeEventListener(name, fn) { this.listeners.get(name)?.delete(fn); }
    fire(name, event={}) { for(const fn of [...(this.listeners.get(name)||[])]) fn(event); }
    getContext() { return {drawImage:(...args)=>this.draws.push(args), clearRect:()=>{this.draws=[];}}; }
    toBlob(fn,type,quality) { h.lastEncoded={width:this.width,height:this.height,type,quality}; fn(new Blob(['jpeg'],{type})); }
    play() { h.playCalls++; return Promise.resolve(); }
    pause() {}
    load() {}
    removeAttribute(name) { delete this[name]; }
  }
  const ids = ['start','stop','source','file-input','file-label','input-source','mode','width','rate','original','result','original-stage','result-stage','connection','message','frame-size','display-fps','round-trip','server-time','frame-count'];
  const elements = Object.fromEntries(ids.map(id=>[id,new Element(id)]));
  elements.mode.value='green'; elements.width.value='640'; elements.rate.value='30'; elements.source.value='camera';
  h.elements=elements;
  const document = new Element('document'); document.hidden=false; document.getElementById=id=>elements[id]; document.createElement=tag=>new Element(tag);
  const window = new Element('window'); window.isSecureContext=true;
  const track = new Element('track'); track.stop=()=>h.trackStops++;
  h.stream={getTracks:()=>[track],getVideoTracks:()=>[track]}; h.track=track;
  h.media=()=>Promise.resolve(h.stream);
  h.bitmap=()=>Promise.resolve({close:()=>h.bitmapCloses++});
  class Socket {
    static OPEN=1; static CLOSING=2;
    constructor(address) { this.url=address; this.readyState=1; this.bufferedAmount=0; this.sent=[]; this.closed=false; h.sockets.push(this); }
    send(value) { this.sent.push(value); }
    close() { this.closed=true; this.readyState=3; }
    json(value) { this.onmessage?.({data:JSON.stringify(value)}); }
    binary() { this.onmessage?.({data:new ArrayBuffer(2)}); }
  }
  class LocalURL extends URL {
    static createObjectURL(blob) { const address='blob:local/'+(h.urls.length+1); h.urls.push(address); return address; }
    static revokeObjectURL(address) { h.revoked.push(address); }
  }
  const context = {document,window,location:new URL(url),navigator:{mediaDevices:{getUserMedia:options=>{h.cameraCalls++; h.cameraOptions=options; return h.media();}}},
    WebSocket:Socket, URL:LocalURL, Blob, ArrayBuffer, performance:{now:()=>h.now}, console,
    createImageBitmap:blob=>h.bitmap(blob),
    requestAnimationFrame:fn=>{const id=h.next++;h.rafs.set(id,fn);return id;}, cancelAnimationFrame:id=>h.rafs.delete(id),
    setTimeout:(fn,ms)=>{const id=h.next++;h.timers.set(id,{fn,ms});return id;},clearTimeout:id=>h.timers.delete(id)};
  vm.runInNewContext(code,context);
  h.document=document; h.window=window;
  h.fire=(id,event)=>elements[id].fire(event);
  h.step=now=>{h.now=now;const pending=[...h.rafs];h.rafs.clear();for(const [,fn] of pending)fn(now);};
  h.start=async()=>{h.fire('start','click');await flush();return h.sockets.at(-1);};
  h.ready=socket=>{socket.json({type:'hello',protocol:1});socket.json({type:'ready'});socket.json({type:'configured'});};
  h.response=(socket,seq=1)=>{socket.json({type:'frame',seq,width:640,height:360,processing_ms:7.5,warmup:false});socket.binary();};
  h.frames=socket=>socket.sent.filter(item=>typeof item!=='string').length;
  return h;
}
"""


CASES = {
    "explicit_permission_and_late_camera": r"""
      const h=browser(), permission=deferred(); h.media=()=>permission.promise;
      assert.equal(h.cameraCalls,0); assert.equal(h.sockets.length,0);
      h.fire('start','click'); assert.equal(h.cameraCalls,1); assert.equal(h.cameraOptions.audio,false);
      h.fire('stop','click'); permission.resolve(h.stream); await flush();
      assert.equal(h.trackStops,1); assert.equal(h.sockets.length,0); assert.equal(h.elements.original.srcObject,null);
    """,
    "one_frame_until_actual_drawing_and_rate_limit": r"""
      const h=browser(), ws=await h.start(); h.ready(ws);
      assert.equal(JSON.parse(ws.sent[0]).max_width,640);
      h.step(0); assert.equal(h.frames(ws),1); assert.equal(h.lastEncoded.width,640); assert.equal(h.lastEncoded.height,360);
      h.step(10); assert.equal(h.frames(ws),1);
      h.response(ws); await flush(); assert.equal(h.frames(ws),1);
      h.step(20); assert.equal(h.elements['frame-count'].textContent,'1');
      assert.equal(h.elements['round-trip'].textContent,'20'); assert.equal(h.elements['server-time'].textContent,'7.5');
      h.step(25); assert.equal(h.frames(ws),1);
      h.step(40); assert.equal(h.frames(ws),2);
      h.response(ws,2); await flush(); h.step(80);
      assert.equal(h.elements['display-fps'].textContent,'16.7'); // actual 60ms drawing interval, not target 30.
      h.fire('stop','click'); assert(ws.closed); assert.equal(h.trackStops,1); assert.equal(h.rafs.size,0); assert.equal(h.timers.size,0);
    """,
    "late_decoded_result_cannot_pollute_new_session": r"""
      const h=browser(), image=deferred(); h.bitmap=()=>image.promise;
      const old=await h.start(); h.ready(old); h.step(0); const lateMessage=old.onmessage; h.response(old); await flush();
      h.fire('stop','click'); const fresh=await h.start(); h.ready(fresh);
      image.resolve({close:()=>h.bitmapCloses++}); await flush();
      lateMessage({data:JSON.stringify({type:'error',message:'stale failure'})});
      assert.equal(h.bitmapCloses,1); assert.equal(h.elements['frame-count'].textContent,'0');
      assert.equal(h.elements.connection.dataset.state,'running'); assert(!fresh.closed);
    """,
    "control_changes_release_resources": r"""
      for(const control of ['mode','width','rate','source','file-input']) {
        const h=browser(),ws=await h.start();h.ready(ws);h.fire(control,'change');
        assert(ws.closed);assert.equal(h.trackStops,1);assert.equal(h.elements.original.srcObject,null);
        assert.equal(h.elements.start.disabled,false);assert.equal(h.rafs.size,0);
      }
    """,
    "background_disconnect_and_camera_ended_release": r"""
      for(const event of ['hidden','close','error','ended','pagehide']) {
        const h=browser(),ws=await h.start();h.ready(ws);
        if(event==='hidden'){h.document.hidden=true;h.document.fire('visibilitychange');}
        if(event==='close')ws.onclose({code:1013});
        if(event==='error')ws.onerror({});
        if(event==='ended')h.track.fire('ended');
        if(event==='pagehide')h.window.fire('pagehide');
        assert(ws.closed);assert.equal(h.trackStops,1);assert.equal(h.timers.size,0);assert.equal(h.rafs.size,0);
      }
    """,
    "permission_denial_and_secure_transport": r"""
      const h=browser();h.media=()=>Promise.reject({name:'NotAllowedError'});await h.start();
      assert.equal(h.sockets.length,0);assert.equal(h.elements.connection.dataset.state,'error');assert(h.elements.message.textContent.includes('权限'));
      const insecure=browser('http://example.invalid:7990/');await insecure.start();
      assert.equal(insecure.cameraCalls,0);assert.equal(insecure.sockets.length,0);
      const secure=browser('https://example.invalid/');const ws=await secure.start();assert.equal(ws.url,'wss://example.invalid/ws');
    """,
    "local_video_is_lazy_frame_only_and_revokes_every_session": r"""
      const h=browser();h.elements.source.value='file';h.fire('source','change');
      h.elements['file-input'].files=[new Blob(['source video'])]; h.fire('file-input','change');
      assert.equal(h.urls.length,0);assert.equal(h.playCalls,0);assert.equal(h.cameraCalls,0);
      const ws=await h.start();h.ready(ws);h.step(0);
      assert.equal(h.cameraCalls,0);assert.equal(h.playCalls,1);assert.equal(h.urls.length,1);assert.equal(h.elements.original.loop,false);
      assert.equal(h.frames(ws),1);assert.equal(ws.sent.length,2);assert(!JSON.stringify(ws.sent[0]).includes('blob:'));
      h.elements.original.fire('ended');assert(ws.closed);assert.deepEqual(h.revoked,h.urls);
      await h.start();assert.equal(h.urls.length,2);h.window.fire('pagehide');assert.deepEqual(h.revoked,h.urls);
      assert.equal(h.elements.original.src,undefined);
    """,
    "local_video_requires_selection_and_malformed_frames_stop": r"""
      const h=browser();h.elements.source.value='file';await h.start();assert.equal(h.sockets.length,0);assert.equal(h.cameraCalls,0);
      const second=browser(),ws=await second.start();second.ready(ws);second.step(0);
      ws.binary();await flush();assert(ws.closed);assert.equal(second.trackStops,1);assert.equal(second.elements.connection.dataset.state,'error');
    """,
    "timeouts_release_camera_and_socket": r"""
      for(const activeFrame of [false,true]){
        const h=browser(),ws=await h.start();if(activeFrame){h.ready(ws);h.step(0);}
        const timeout=[...h.timers.values()].find(item=>item.ms===(activeFrame?60000:120000));timeout.fn();
        assert(ws.closed);assert.equal(h.trackStops,1);assert.equal(h.elements.start.disabled,false);
      }
    """,
}


@pytest.mark.parametrize("scenario", list(CASES))
def test_controller_lifecycle(scenario):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional; install it to run browser-controller unit tests")
    script = HARNESS + "\n(async()=>{\n" + CASES[scenario] + "\n})().catch(error=>{console.error(error);process.exitCode=1;});"
    completed = subprocess.run([node, "-e", script, str(FRONTEND)], capture_output=True,
                               text=True, encoding="utf-8", timeout=15)
    assert completed.returncode == 0, completed.stdout + completed.stderr
