import { Field3 } from '../src/core/grid.js';
import { dualContour, largestComponent } from '../src/core/mesher.js';
// unit sphere r=0.5, phi<0 inside
const f=new Field3(40,1.0); f.fill((x,y,z)=>Math.hypot(x,y,z)-0.5);
let m=largestComponent(dualContour(f,{sharpness:0.6}));
const P=m.positions,I=m.indices;
let vol=0, agree=0, tot=0;
for(let t=0;t<I.length;t+=3){
  const a=I[t]*3,b=I[t+1]*3,c=I[t+2]*3;
  const ax=P[a],ay=P[a+1],az=P[a+2],bx=P[b],by=P[b+1],bz=P[b+2],cx=P[c],cy=P[c+1],cz=P[c+2];
  vol+=(ax*(by*cz-bz*cy)-ay*(bx*cz-bz*cx)+az*(bx*cy-by*cx))/6;
  // face normal vs outward radial
  const e1=[bx-ax,by-ay,bz-az],e2=[cx-ax,cy-ay,cz-az];
  const nx=e1[1]*e2[2]-e1[2]*e2[1],ny=e1[2]*e2[0]-e1[0]*e2[2],nz=e1[0]*e2[1]-e1[1]*e2[0];
  const cxx=(ax+bx+cx)/3,cyy=(ay+by+cy)/3,czz=(az+bz+cz)/3;
  if(nx*cxx+ny*cyy+nz*czz>0) agree++; tot++;
}
console.log('signed volume',vol.toFixed(4),'(true 0.5236)');
console.log('faces pointing OUTWARD:',agree,'/',tot,(100*agree/tot).toFixed(1)+'%');
// gradient normals from DC
let ga=0; for(let v=0;v<P.length;v+=3){ if(m.normals[v]*P[v]+m.normals[v+1]*P[v+1]+m.normals[v+2]*P[v+2]>0) ga++; }
console.log('DC gradient normals outward:',ga,'/',P.length/3);

// recomputeNormals must agree with the (now correct) winding
import { recomputeNormals } from '../src/core/mesher.js';
recomputeNormals(m);
let ra=0; const N=m.normals;
for(let v=0;v<P.length;v+=3) if(N[v]*P[v]+N[v+1]*P[v+1]+N[v+2]*P[v+2]>0) ra++;
console.log('recomputeNormals outward:', ra,'/',P.length/3);
if (vol<0 || agree!==tot || ra!==P.length/3) { console.log('FAIL'); process.exit(1); }
console.log('PASS');
