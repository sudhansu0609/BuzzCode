//! BUZZCODE ARCADE (graphical): a tiny localhost HTTP server that serves a canvas page and
//! streams the live factory state to it over SSE. No external crates: hand-rolled HTTP/1.1.

use crate::factory::{Factory, WorkerState};
use anyhow::Result;
use serde::Serialize;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::sync::watch;

#[derive(Serialize)]
struct WorkerJson<'a> { id: u64, name: &'a str, role: &'a str, task: &'a str, state: &'static str, activity: &'a str, tools: u32, turns: u32, secs: u64, loot: String, errors: u32, active: bool }

#[derive(Serialize)]
struct Snapshot<'a> { mission: &'a str, score: u64, hi: u64, level: u32, secs: u64, lives: u32, model: &'a str, tps: f64, ctx_used: u32, ctx_total: u32, busy: bool, workers: Vec<WorkerJson<'a>> }

fn state_name(s: WorkerState) -> &'static str {
    match s { WorkerState::Idle => "idle", WorkerState::Thinking => "thinking", WorkerState::Writing => "writing", WorkerState::Tool => "tool", WorkerState::Waiting => "waiting", WorkerState::Done => "done", WorkerState::Failed => "failed" }
}

pub fn snapshot_json(f: &Factory, model: &str, tps: f64, ctx_used: u32, ctx_total: u32, busy: bool) -> String {
    let p1 = &f.workers[0];
    let secs = p1.finished.unwrap_or_else(Instant::now).duration_since(p1.started).as_secs();
    let errors: u32 = f.workers.iter().map(|w| w.errors).sum();
    let snap = Snapshot {
        mission: &f.big_task, score: f.score(), hi: f.hi_score.max(f.score()), level: p1.turns + 1, secs,
        lives: 3u32.saturating_sub(errors / 3), model, tps, ctx_used, ctx_total, busy,
        workers: f.workers.iter().map(|w| WorkerJson {
            id: w.id, name: &w.name, role: &w.role, task: &w.task, state: state_name(w.state), activity: &w.activity,
            tools: w.tools_used, turns: w.turns, secs: w.finished.unwrap_or_else(Instant::now).duration_since(w.started).as_secs(),
            loot: w.loot.iter().collect(), errors: w.errors, active: w.id == f.active,
        }).collect(),
    };
    serde_json::to_string(&snap).unwrap_or_else(|_| "{}".into())
}

pub struct ArcadeServer {
    pub port: u16,
    tx: watch::Sender<String>,
}

impl ArcadeServer {
    /// Bind on 127.0.0.1:`port` (or the next free port) and serve forever in the background.
    pub async fn start(port: u16) -> Result<Self> {
        let (tx, rx) = watch::channel(String::from("{}"));
        let mut bound = None;
        for p in port..port + 20 {
            if let Ok(l) = TcpListener::bind(("127.0.0.1", p)).await { bound = Some((p, l)); break; }
        }
        let (port, listener) = bound.ok_or_else(|| anyhow::anyhow!("no free port for the arcade server"))?;
        tokio::spawn(async move {
            loop {
                let Ok((sock, _)) = listener.accept().await else { continue };
                let rx = rx.clone();
                tokio::spawn(async move { let _ = handle(sock, rx).await; });
            }
        });
        Ok(Self { port, tx })
    }

    pub fn publish(&self, json: String) { let _ = self.tx.send(json); }
    pub fn url(&self) -> String { format!("http://127.0.0.1:{}/", self.port) }

    /// Open the page in the default browser (Windows `start`, else xdg-open/open).
    pub fn open_browser(&self) {
        let url = self.url();
        #[cfg(windows)]
        { let _ = std::process::Command::new("cmd").args(["/C", "start", "", &url]).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).spawn(); }
        #[cfg(target_os = "macos")]
        { let _ = std::process::Command::new("open").arg(&url).spawn(); }
        #[cfg(all(unix, not(target_os = "macos")))]
        { let _ = std::process::Command::new("xdg-open").arg(&url).spawn(); }
    }
}

async fn handle(sock: tokio::net::TcpStream, mut rx: watch::Receiver<String>) -> std::io::Result<()> {
    let (r, mut w) = sock.into_split();
    let mut reader = BufReader::new(r);
    let mut line = String::new();
    reader.read_line(&mut line).await?;
    let path = line.split_whitespace().nth(1).unwrap_or("/").to_string();
    // Drain headers.
    loop { let mut h = String::new(); if reader.read_line(&mut h).await? == 0 || h == "\r\n" || h == "\n" { break; } }
    match path.as_str() {
        "/events" => {
            w.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n").await?;
            // Send current state immediately, then on every change (coalesced), plus keepalives.
            let mut last = rx.borrow().clone();
            w.write_all(format!("data: {last}\n\n").as_bytes()).await?;
            loop {
                tokio::select! {
                    changed = rx.changed() => {
                        if changed.is_err() { break; }
                        let cur = rx.borrow().clone();
                        if cur != last { w.write_all(format!("data: {cur}\n\n").as_bytes()).await?; last = cur; }
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_secs(15)) => { w.write_all(b": ping\n\n").await?; }
                }
            }
            Ok(())
        }
        "/" | "/index.html" => {
            let body = PAGE.as_bytes();
            w.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).as_bytes()).await?;
            w.write_all(body).await?;
            w.shutdown().await
        }
        _ => { w.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await?; w.shutdown().await }
    }
}

const PAGE: &str = r##"<!doctype html>
<html><head><meta charset="utf-8"><title>BUZZCODE ARCADE</title>
<style>
  html,body{margin:0;height:100%;background:#07040f;overflow:hidden;font-family:"Segoe UI",system-ui,sans-serif}
  canvas{display:block;width:100vw;height:100vh;image-rendering:pixelated}
  #off{position:fixed;inset:0;display:none;align-items:center;justify-content:center;color:#ff3fc8;font:bold 28px monospace;background:#07040fcc;letter-spacing:.1em}
</style></head><body>
<canvas id="c"></canvas><div id="off">■ CONNECTION LOST — is buzzcode running? ■</div>
<script>
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
let S={mission:'',score:0,hi:0,level:1,secs:0,lives:3,model:'',tps:0,ctx_used:0,ctx_total:1,busy:false,workers:[]};
let t=0,particles=[],confetti=[],shake=0,lastState={},flash=0,prevScore=0,pop=[];
const COLORS={idle:'#6b7280',thinking:'#22d3ee',writing:'#60a5fa',tool:'#facc15',waiting:'#e879f9',done:'#4ade80',failed:'#f87171'};
const LOOT={'▤':'#60a5fa','◎':'#22d3ee','◆':'#facc15','⚙':'#fb923c','★':'#f472b6','▣':'#a78bfa','♦':'#34d399','●':'#e5e7eb'};
const LOOTNAME={'▤':'READ','◎':'SEARCH','◆':'EDIT','⚙':'SHELL','★':'SPAWN','▣':'PLAN','♦':'GIT','●':'TOOL'};
// 12x14 pixel sprite maps (0 bg, 1 outline, 2 skin, 3 suit, 4 visor/accent)
const BODY=[
"000111111000","001222222100","012244442210","012244442210","012222222210","001222222100","000133331000",
"001333333100","013333333310","013313313310","001333333100","000133331000","000131131000","000111111000"];
const LEGS=[["000131131000","000111111000"],["001310013100","001110011100"]];
function resize(){cv.width=innerWidth;cv.height=innerHeight}addEventListener('resize',resize);resize();
function px(x,y,s,c){ctx.fillStyle=c;ctx.fillRect(Math.round(x),Math.round(y),Math.ceil(s),Math.ceil(s))}
function drawSprite(x,y,s,col,frame,state){
  const pal={1:'#140a22',2:'#ffd5a8',3:col,4:state==='done'?'#fff7ae':state==='failed'?'#ff5c5c':'#d9fbff'};
  for(let r=0;r<12;r++)for(let c=0;c<12;c++){const v=+BODY[r][c];if(v)px(x+c*s,y+r*s,s,pal[v]);}
  const legs=LEGS[frame%2];for(let r=0;r<2;r++)for(let c=0;c<12;c++){const v=+legs[r][c];if(v)px(x+c*s,y+(12+r)*s,s,pal[v]);}
  if(state==='tool'){ // hammer
    const hx=x+13*s,hy=y+(frame%2?3:6)*s;px(hx,hy,s*2,'#9ca3af');px(hx+s*2,hy,s,'#9ca3af');px(hx+s,hy+s*2,s,'#a16207');px(hx+s,hy+s*3,s,'#a16207');
  } else if(state==='writing'){px(x+13*s,y+7*s,s,'#fde68a');px(x+14*s,y+6*s,s,'#fde68a');px(x+15*s,y+5*s,s,'#f87171');}
  else if(state==='thinking'){const b=y-4*s-Math.sin(t/10)*s;px(x+12*s,y-s,s,'#fff');px(x+13*s,b+3*s,s*2,'#fff');ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(x+15*s,b,4*s,0,7);ctx.fill();ctx.fillStyle='#22d3ee';ctx.font=`bold ${5*s}px monospace`;ctx.fillText('?',x+13.3*s,b+1.8*s);}
  else if(state==='waiting'){px(x-s,y+(frame%2?4:5)*s,s,'#ffd5a8');px(x+12*s,y+(frame%2?4:5)*s,s,'#ffd5a8');px(x-2*s,y+3*s,s,'#ffd5a8');px(x+13*s,y+3*s,s,'#ffd5a8');}
  else if(state==='done'){ctx.fillStyle='#fde047';ctx.font=`bold ${6*s}px monospace`;ctx.fillText('★',x+12*s,y+2*s-Math.abs(Math.sin(t/8))*2*s);}
}
function bubble(x,y,text,col,maxw){
  ctx.font='bold 15px "Segoe UI",sans-serif';let tw=Math.min(ctx.measureText(text).width+20,maxw);
  ctx.fillStyle='#0b0618ee';ctx.strokeStyle=col;ctx.lineWidth=2;rr(x,y-34,tw,28,8);ctx.fill();ctx.stroke();
  ctx.fillStyle='#fff';ctx.save();ctx.beginPath();ctx.rect(x,y-34,tw,28);ctx.clip();ctx.fillText(text,x+10,y-14);ctx.restore();
  ctx.fillStyle=col;ctx.beginPath();ctx.moveTo(x+18,y-6);ctx.lineTo(x+26,y-6);ctx.lineTo(x+22,y+2);ctx.fill();
}
function rr(x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath()}
function hud(){
  const W=cv.width;ctx.fillStyle='#0b0618';ctx.fillRect(0,0,W,64);
  ctx.fillStyle='#ff3fc8';ctx.fillRect(0,64,W,3);
  ctx.font='bold 22px monospace';ctx.fillStyle='#fde047';ctx.fillText('LV'+String(S.level).padStart(2,'0'),20,40);
  ctx.fillStyle='#fff';ctx.fillText('SCORE '+String(S.score).padStart(6,'0'),120,40);
  ctx.fillStyle='#a78bfa';ctx.fillText('HI '+String(S.hi).padStart(6,'0'),380,40);
  const m=Math.floor(S.secs/60),s=S.secs%60;ctx.fillStyle='#22d3ee';ctx.fillText(String(m).padStart(2,'0')+':'+String(s).padStart(2,'0'),600,40);
  for(let i=0;i<3;i++){ctx.fillStyle=i<S.lives?'#f87171':'#3f1d2e';ctx.font='26px monospace';ctx.fillText('♥',720+i*30,42);}
  ctx.font='bold 16px monospace';ctx.fillStyle='#4ade80';ctx.textAlign='right';
  ctx.fillText((S.tps?S.tps.toFixed(1):'–')+' tok/s  ·  ctx '+Math.round(S.ctx_used/1000)+'k/'+Math.round(S.ctx_total/1000)+'k  ·  '+S.model,W-20,40);ctx.textAlign='left';
  ctx.font='bold 18px "Segoe UI",sans-serif';ctx.fillStyle='#fff';
  const mission=S.mission?('MISSION: '+S.mission):((Math.floor(t/20)%2)?'INSERT COIN  ▮':'INSERT COIN');
  ctx.save();ctx.beginPath();ctx.rect(0,70,W,40);ctx.clip();ctx.fillText(mission,20,98);ctx.restore();
}
function background(){
  const W=cv.width,H=cv.height;const g=ctx.createLinearGradient(0,0,0,H);g.addColorStop(0,'#12082a');g.addColorStop(.6,'#2a0b4a');g.addColorStop(1,'#3b0f5c');ctx.fillStyle=g;ctx.fillRect(0,0,W,H);
  // stars
  for(let i=0;i<70;i++){const x=(i*173+Math.floor(t/3)*0)%W,y=110+(i*97)%(H*0.4);ctx.fillStyle=(i+Math.floor(t/15))%7?'#ffffff55':'#ffffffcc';ctx.fillRect(x,y,2,2);}
  // parallax skyline
  for(let L=0;L<2;L++){const sp=(L+1)*0.4,base=H*(0.55+L*0.08);ctx.fillStyle=L?'#1b0b35':'#120728';for(let i=0;i<40;i++){const w=60+(i*37)%90,h=60+(i*53)%140,x=((i*140- t*sp)%(W+200)+W+200)%(W+200)-100;ctx.fillRect(x,base-h,w,h);ctx.fillStyle=L?'#fde04733':'#22d3ee22';for(let wy=base-h+10;wy<base-8;wy+=16)for(let wx=x+8;wx<x+w-8;wx+=14)if((wx*7+wy*3+i)%5<2)ctx.fillRect(wx,wy,6,8);ctx.fillStyle=L?'#1b0b35':'#120728';}}
}
function lane(w,i,n){
  const W=cv.width,H=cv.height,top=120,avail=H-top-20,lh=Math.min(avail/n,260),y=top+i*lh,s=Math.max(3,Math.min(6,Math.floor(lh/40)));
  const col=COLORS[w.state]||'#fff';const floor=y+lh-30;
  // conveyor floor
  ctx.fillStyle='#1f1235';ctx.fillRect(0,floor,W,16);ctx.fillStyle=w.state==='done'?'#4ade80':'#22d3ee';
  const off=w.active?(t*3)%40:0;for(let x=-40+off;x<W;x+=40)ctx.fillRect(x,floor+2,24,4);
  ctx.fillStyle='#ff3fc8';ctx.fillRect(0,floor+16,W,2);
  // name plate
  ctx.font='bold 20px monospace';ctx.fillStyle=w.active?'#fde047':col;ctx.fillText((w.active?'▶ ':'  ')+'P'+(i+1)+' '+w.name,20,y+28);
  ctx.font='16px monospace';ctx.fillStyle='#c4b5fd';ctx.fillText('['+w.role+']',20+ctx.measureText('▶ P'+(i+1)+' '+w.name).width+40,y+28);
  ctx.fillStyle='#9ca3af';ctx.fillText('⚒'+w.tools+'  LV'+w.turns+'  ✗'+w.errors+'  '+w.secs+'s',20,y+50);
  if(w.task&&w.id!==0){ctx.fillStyle='#d1d5db';ctx.font='italic 15px "Segoe UI",sans-serif';ctx.save();ctx.beginPath();ctx.rect(20,y+56,W-40,22);ctx.clip();ctx.fillText('brief: '+w.task,20,y+72);ctx.restore();}
  // player position: advances with tools, jogs while active
  const trackW=W-260,run=(w.tools*60+(w.active&&S.busy?t*1.2:0))%trackW;const px0=200+run,py=floor-14*s;
  const frame=(w.active&&S.busy)?Math.floor(t/6):0;
  // loot trail
  const loot=[...w.loot];for(let k=0;k<Math.min(loot.length,14);k++){const g=loot[loot.length-1-k];const lx=px0-(k+1)*34,ly=floor-22-Math.sin((t+k*9)/12)*4;if(lx<200)break;ctx.fillStyle=LOOT[g]||'#fff';rr(lx,ly-10,22,22,5);ctx.fill();ctx.fillStyle='#0b0618';ctx.font='bold 10px monospace';ctx.textAlign='center';ctx.fillText(LOOTNAME[g]?.slice(0,3)||'T',lx+11,ly+4);ctx.textAlign='left';}
  // shadow + sprite
  ctx.fillStyle='#00000066';ctx.beginPath();ctx.ellipse(px0+6*s,floor+2,9*s,3,0,0,7);ctx.fill();
  drawSprite(px0,py,s,col,frame,w.state);
  bubble(px0+10*s,py-6,w.activity||'',col,Math.min(420,W-px0-40));
}
function banner(){
  const p1=S.workers[0];if(!p1)return;const W=cv.width,H=cv.height;let txt='',col='#fde047';
  if(p1.state==='done'){txt='★ STAGE CLEAR ★';col='#4ade80'}else if(p1.state==='failed'){txt='GAME OVER — CONTINUE? (send a message)';col='#f87171'}else if(p1.state==='idle'&&S.mission){txt='PLAYER 1 READY'}
  if(!txt||Math.floor(t/25)%2)return;ctx.font='bold 36px monospace';ctx.textAlign='center';ctx.fillStyle='#000a';ctx.fillText(txt,W/2+3,H-40+3);ctx.fillStyle=col;ctx.fillText(txt,W/2,H-40);ctx.textAlign='left';
}
function fx(){
  particles=particles.filter(p=>p.life>0);for(const p of particles){p.x+=p.vx;p.y+=p.vy;p.vy+=0.15;p.life--;ctx.fillStyle=p.c;ctx.globalAlpha=Math.max(0,p.life/30);ctx.fillRect(p.x,p.y,p.s,p.s);}ctx.globalAlpha=1;
  confetti=confetti.filter(c=>c.y<cv.height+20);for(const c of confetti){c.y+=c.vy;c.x+=Math.sin((t+c.p)/7)*1.5;ctx.fillStyle=c.c;ctx.fillRect(c.x,c.y,6,10);}
  pop=pop.filter(p=>p.life>0);for(const p of pop){p.y-=1.2;p.life--;ctx.globalAlpha=p.life/40;ctx.font='bold 22px monospace';ctx.fillStyle='#fde047';ctx.fillText(p.txt,p.x,p.y);}ctx.globalAlpha=1;
  if(flash>0){ctx.fillStyle=`rgba(255,255,255,${flash/10})`;ctx.fillRect(0,0,cv.width,cv.height);flash--;}
}
function burst(x,y,c,n){for(let i=0;i<n;i++)particles.push({x,y,vx:(Math.random()-.5)*6,vy:-Math.random()*5-1,s:3+Math.random()*4,c,life:30+Math.random()*20});}
function diff(){
  // compare with last state for effects
  for(const w of S.workers){const o=(lastState.workers||[]).find(v=>v.id===w.id);if(!o)continue;
    if(w.tools>o.tools){const i=S.workers.indexOf(w);const lh=Math.min((cv.height-140)/S.workers.length,260);const y=120+i*lh+lh-60;burst(220+(w.tools*60)%(cv.width-260),y,COLORS.tool,18);pop.push({x:240+(w.tools*60)%(cv.width-260),y:y-20,txt:'+100',life:40});}
    if(w.state==='done'&&o.state!=='done'){for(let i=0;i<120;i++)confetti.push({x:Math.random()*cv.width,y:-20-Math.random()*300,vy:2+Math.random()*3,c:['#ff3fc8','#22d3ee','#fde047','#4ade80','#a78bfa'][i%5],p:i});flash=6;}
    if(w.errors>o.errors){shake=10;}
  }
  if(S.score>prevScore&&prevScore){}prevScore=S.score;lastState=JSON.parse(JSON.stringify(S));
}
function frame(){t++;ctx.save();if(shake>0){ctx.translate((Math.random()-.5)*8,(Math.random()-.5)*8);shake--;}
  background();hud();const ws=S.workers.length?S.workers:[{id:0,name:'FOREMAN',role:'main',state:'idle',activity:'press start',tools:0,turns:0,errors:0,secs:0,loot:'',active:true}];
  ws.forEach((w,i)=>lane(w,i,ws.length));fx();banner();ctx.restore();requestAnimationFrame(frame);}
function connect(){const es=new EventSource('/events');es.onmessage=e=>{try{const s=JSON.parse(e.data);if(s&&s.workers){S=s;diff();document.getElementById('off').style.display='none';}}catch(_){}};es.onerror=()=>{document.getElementById('off').style.display='flex';};}
connect();frame();
</script></body></html>"##;
