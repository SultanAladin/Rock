/** Verify the GLSL hash and the JS hash agree bit-for-bit by emulating GLSL
 *  uint semantics (wrapping 32-bit mul/xor/shift) in BigInt. */
import { hashU32, hash3i } from '../src/core/rng.js';
const M=0xffffffffn;
const hU=(x)=>{x=BigInt(x)&M; x^=x>>16n; x=(x*0x7feb352dn)&M; x^=x>>15n; x=(x*0x846ca68bn)&M; x^=x>>16n; return x;};
const h3=(x,y,z,s)=>{ // GLSL: uint(p.x)*A ^ uint(p.y)*B ^ uint(p.z)*C ^ seed*D
  const u=(v)=>BigInt.asUintN(32,BigInt(v));
  let h=((u(x)*0x8da6b343n)&M) ^ ((u(y)*0xd8163841n)&M) ^ ((u(z)*0xcb1ab31fn)&M) ^ ((u(s)*0x165667b1n)&M);
  return hU(h&M);
};
let bad=0,n=0;
for(let i=-40;i<40;i+=3)for(let j=-40;j<40;j+=7)for(let k=-40;k<40;k+=11){
  const a=BigInt(hash3i(i,j,k,12345)>>>0), b=h3(i,j,k,12345); n++;
  if(a!==b){ if(bad<5) console.log('MISMATCH',i,j,k,a,b); bad++; }
}
console.log('hash3i compared',n,'mismatches',bad);
let b2=0; for(let x=0;x<100000;x+=997){ if(BigInt(hashU32(x)>>>0)!==hU(x)) b2++; }
console.log('hashU32 mismatches',b2);
process.exit(bad||b2?1:0);
