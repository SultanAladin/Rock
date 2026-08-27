/**
 * WGSL rules that wgsl_reflect's parser accepts but a real compiler rejects.
 *
 * wgsl_reflect is permissive: it parsed both of the errors below happily, they
 * reached the browser, and every pipeline silently failed to compile. These are
 * the two classes that actually shipped, plus the neighbouring rules in the
 * same families.
 *
 *   1. MIXED OPERATOR PRECEDENCE. WGSL defines NO relative precedence between
 *      bitwise (& | ^) and arithmetic/shift/relational operators, and requires
 *      explicit parentheses. C, GLSL and JS all bind * tighter than ^, so a
 *      direct port of a hash function is exactly where this bites.
 *
 *   2. RESERVED KEYWORDS. WGSL reserves a large vocabulary for future use --
 *      including ordinary-looking identifiers like `target`, `sample`, `filter`
 *      and `shared` -- and using one as a variable name is a hard error.
 */

import * as ERODE from '../src/gpu/wgsl/erode.wgsl.js';
import { RAYMARCH_WGSL } from '../src/gpu/wgsl/raymarch.wgsl.js';

// WGSL reserved words (spec 'Reserved Words'), minus those already keywords.
const RESERVED = new Set(`
NULL Self abstract active alignas alignof as asm asm_fragment async attribute
auto await become binding_array cast catch class co_await co_return co_yield
coherent column_major common compile compile_fragment concept
const_cast consteval constexpr constinit crate debugger decltype delete demote
demote_to_helper do dynamic_cast enum explicit export extends extern external
fallthrough filter final finally friend from fxgroup get goto groupshared
highp impl implements import inline instanceof interface layout lowp macro
macro_rules match mediump meta mod module move mutable namespace new nil
noexcept noinline nointerpolation non_coherent noncoherent noperspective null
nullptr of operator package packoffset partition pass patch pixelfragment
precise precision premerge priv protected pub public readonly ref regardless
register reinterpret_cast require resource restrict self set shared sizeof
smooth snorm static static_assert static_cast std subroutine super target
template this thread_local throw trait try type typedef typeid typename
typeof union unless unorm unsafe unsized use using varying virtual volatile
wgsl where with writeonly yield
`.trim().split(/\s+/));

// Operator families that WGSL refuses to mix without parentheses.
const BITWISE = ['&', '|', '^'];
const ARITH = ['*', '/', '%', '+', '-', '<<', '>>'];

let fail = 0;

function lint(name, src) {
  const problems = [];
  // strip comments and attributes so prose and @group(0) are not scanned
  const clean = src
    .replace(/\/\*[\s\S]*?\*\//g, (m) => m.replace(/[^\n]/g, ' '))
    .replace(/\/\/[^\n]*/g, '')
    .replace(/@[A-Za-z_]\w*(\s*\([^)]*\))?/g, ' ');
  const lines = clean.split('\n');

  lines.forEach((line, i) => {
    // ---- reserved identifiers ------------------------------------------
    for (const m of line.matchAll(/\b(?:let|var|const)\s+([A-Za-z_]\w*)/g)) {
      if (RESERVED.has(m[1])) {
        problems.push(`${i + 1}: "${m[1]}" is a WGSL reserved keyword\n      ${line.trim()}`);
      }
    }
    for (const m of line.matchAll(/\bfn\s+([A-Za-z_]\w*)/g)) {
      if (RESERVED.has(m[1])) problems.push(`${i + 1}: function name "${m[1]}" is reserved`);
    }
    // struct members and fn parameters
    for (const m of line.matchAll(/([A-Za-z_]\w*)\s*:\s*(?:vec\d|mat\d|f32|i32|u32|bool|array|ptr|atomic)/g)) {
      if (RESERVED.has(m[1])) problems.push(`${i + 1}: member/param "${m[1]}" is reserved`);
    }

    // ---- mixed bitwise / arithmetic without parentheses -----------------
    // Per STATEMENT (a line may hold several), keep only the right-hand side,
    // and blank out anything nested inside parentheses/brackets -- WGSL only
    // objects when the two families meet at the SAME paren depth.
    const segs = [];
    for (const stmt of line.split(';')) {
      const eq = stmt.search(/[^<>=!*/+\-^&|]=[^=]/);
      const expr = eq >= 0 ? stmt.slice(eq + 2) : stmt;
      let depth = 0, seg = '';
      for (const ch of expr) {
        if (ch === '(' || ch === '[') { depth++; continue; }
        if (ch === ')' || ch === ']') { depth--; continue; }
        seg += depth === 0 ? ch : ' ';
      }
      segs.push(seg);
    }
    for (const s2 of segs) {
      // ignore ^= etc and template brackets
      const t = s2.replace(/[<>]/g, ' ');
      const hasBit = BITWISE.some((o) => {
        const idx = t.indexOf(o);
        return idx >= 0 && t[idx + 1] !== '=' && t[idx - 1] !== o && t[idx + 1] !== o;
      });
      if (!hasBit) continue;
      const hasArith = ARITH.some((o) => {
        let idx = t.indexOf(o);
        while (idx >= 0) {
          const nxt = t[idx + 1], prv = t[idx - 1];
          // skip ->, =-, unary minus after an operator, and float exponents
          if (!(o === '-' && (nxt === '>' || /[=*/+\-^&|,(]\s*$/.test(t.slice(0, idx))))) {
            if (nxt !== '=') return true;
          }
          idx = t.indexOf(o, idx + 1);
        }
        return false;
      });
      if (hasArith) {
        problems.push(
          `${i + 1}: mixes bitwise and arithmetic operators at the same paren depth; ` +
          `WGSL requires explicit parentheses\n      ${line.trim()}`);
      }
    }
  });

  if (problems.length) {
    console.log(`[FAIL] ${name}`);
    for (const p of [...new Set(problems)]) console.log(`    ${p}`);
    fail += problems.length;
  } else {
    console.log(`[ OK ] ${name}`);
  }
}

const mods = Object.entries(ERODE)
  .filter(([k, v]) => k.endsWith('_WGSL') && typeof v === 'string')
  .concat([['RAYMARCH_WGSL', RAYMARCH_WGSL]]);
for (const [k, v] of mods) lint(k.replace('_WGSL', ''), v);

console.log(fail ? `\nFAIL (${fail})` : '\nPASS');
process.exit(fail ? 1 : 0);
