// 연구노트용 그림 생성기. JSON 스펙 -> SVG (+ 크롬이 있으면 PNG)
// 사용: node make_figure.cjs <spec.json> <out.svg> [--png]
// 스펙 종류
//   { "type": "bar",    "title": "...", "unit": "%", "items": [{ "label": "합성", "value": 98.2 }, { "label": "실데이터", "value": 24.0, "highlight": true }] }
//   { "type": "change", "title": "...", "unit": "%", "legend": ["수정 전", "수정 후"], "items": [{ "label": "자기검증", "from": 42.9, "to": 100 }] }
//   { "type": "flow",   "title": "...", "steps": ["질의", "생성", "검증"], "back": { "from": 2, "to": 1, "label": "미달 시 재생성" } }
// 규칙: 강조 1색(파랑) + 회색, 값은 막대 끝에 직접 표기, 격자선과 테두리 없음, 축은 하나
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const C = { ink: '#0b0b0b', sub: '#52514e', mute: '#c9c8c2', main: '#2a78d6', bg: '#ffffff', box: '#f1f0ec' };
const FONT = "'Malgun Gothic','맑은 고딕',sans-serif";
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const fmt = (v, unit) => (Math.abs(v) >= 10000 ? v.toLocaleString('en-US') : String(v)) + (unit || '');
// 한글은 대략 1em, 영문과 숫자는 0.56em 으로 폭 추정
const tw = (s, size) => [...String(s)].reduce((a, ch) => a + (/[\u3131-\uD7A3]/.test(ch) ? 1 : 0.56), 0) * size;

function head(spec, W) {
  let y = 30;
  let out = `<text x="24" y="${y}" font-size="17" font-weight="700" fill="${C.ink}">${esc(spec.title || '')}</text>`;
  if (spec.note) {
    y += 22;
    out += `<text x="24" y="${y}" font-size="12.5" fill="${C.sub}">${esc(spec.note)}</text>`;
  }
  return { svg: out, y: y + 26 };
}

function wrap(W, H, body) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="${FONT}"><rect width="${W}" height="${H}" fill="${C.bg}"/>${body}</svg>`;
}

function bar(spec) {
  const W = 720, rowH = 40, barH = 20;
  const h = head(spec, W);
  const labelW = Math.max(...spec.items.map((i) => tw(i.label, 13))) + 16;
  const x0 = 24 + labelW;
  const max = spec.max || Math.max(...spec.items.map((i) => i.value));
  const span = W - x0 - 90;
  let body = h.svg;
  spec.items.forEach((it, i) => {
    const y = h.y + i * rowH;
    const w = Math.max(2, (it.value / max) * span);
    const fill = it.highlight ? C.main : C.mute;
    body += `<text x="${x0 - 12}" y="${y + barH / 2 + 4.5}" font-size="13" text-anchor="end" fill="${C.ink}">${esc(it.label)}</text>`;
    body += `<path d="M${x0},${y} h${w - 4} a4,4 0 0 1 4,4 v${barH - 8} a4,4 0 0 1 -4,4 h-${w - 4} z" fill="${fill}"/>`;
    body += `<text x="${x0 + w + 8}" y="${y + barH / 2 + 4.5}" font-size="13" font-weight="${it.highlight ? 700 : 400}" fill="${C.ink}">${esc(fmt(it.value, spec.unit))}</text>`;
  });
  return wrap(W, h.y + spec.items.length * rowH + 8, body);
}

function change(spec) {
  const W = 720, rowH = 46;
  const h = head(spec, W);
  const labelW = Math.max(...spec.items.map((i) => tw(i.label, 13))) + 16;
  const x0 = 24 + labelW + 50;
  const all = spec.items.flatMap((i) => [i.from, i.to]);
  const max = spec.max || Math.max(...all), min = spec.min ?? Math.min(0, ...all);
  const span = W - x0 - 80;
  const X = (v) => x0 + ((v - min) / (max - min || 1)) * span;
  let body = h.svg;
  body += `<circle cx="${W - 196}" cy="24" r="5" fill="${C.mute}"/><text x="${W - 186}" y="28.5" font-size="12" fill="${C.sub}">${esc((spec.legend || [])[0] || '수정 전')}</text>`;
  body += `<circle cx="${W - 112}" cy="24" r="5" fill="${C.main}"/><text x="${W - 102}" y="28.5" font-size="12" fill="${C.sub}">${esc((spec.legend || [])[1] || '수정 후')}</text>`;
  spec.items.forEach((it, i) => {
    const y = h.y + i * rowH + 10;
    const a = X(it.from), b = X(it.to);
    const u = it.unit ?? spec.unit;
    body += `<text x="${24 + labelW - 12}" y="${y + 4.5}" font-size="13" text-anchor="end" fill="${C.ink}">${esc(it.label)}</text>`;
    body += `<line x1="${a}" y1="${y}" x2="${b}" y2="${y}" stroke="${C.mute}" stroke-width="2"/>`;
    body += `<circle cx="${a}" cy="${y}" r="5" fill="${C.mute}" stroke="${C.bg}" stroke-width="2"/>`;
    body += `<circle cx="${b}" cy="${y}" r="6" fill="${C.main}" stroke="${C.bg}" stroke-width="2"/>`;
    const left = Math.min(a, b), right = Math.max(a, b);
    const lv = a <= b ? it.from : it.to, rv = a <= b ? it.to : it.from;
    body += `<text x="${left - 10}" y="${y + 4.5}" font-size="12.5" text-anchor="end" fill="${C.sub}" font-weight="${a <= b ? 400 : 700}">${esc(fmt(lv, u))}</text>`;
    body += `<text x="${right + 11}" y="${y + 4.5}" font-size="12.5" fill="${C.ink}" font-weight="${a <= b ? 700 : 400}">${esc(fmt(rv, u))}</text>`;
  });
  return wrap(W, h.y + spec.items.length * rowH + 4, body);
}

function flow(spec) {
  const W = 720, boxH = 40, gap = 28, pad = 24;
  const h = head(spec, W);
  const widths = spec.steps.map((s) => Math.max(72, tw(s, 13) + 28));
  const rows = [[]];
  let x = pad;
  widths.forEach((w, i) => {
    if (x + w > W - pad && rows[rows.length - 1].length) { rows.push([]); x = pad; }
    rows[rows.length - 1].push({ i, x, w });
    x += w + gap;
  });
  let body = h.svg;
  const pos = {};
  rows.forEach((row, r) => {
    const y = h.y + r * (boxH + 34);
    row.forEach((b, k) => {
      pos[b.i] = { x: b.x, y, w: b.w };
      body += `<rect x="${b.x}" y="${y}" width="${b.w}" height="${boxH}" rx="6" fill="${C.box}"/>`;
      body += `<text x="${b.x + b.w / 2}" y="${y + boxH / 2 + 4.5}" font-size="13" text-anchor="middle" fill="${C.ink}">${esc(spec.steps[b.i])}</text>`;
      if (k < row.length - 1) {
        const ax = b.x + b.w + 5, ay = y + boxH / 2;
        body += `<path d="M${ax},${ay} h${gap - 12} m-5,-4 l5,4 l-5,4" stroke="${C.sub}" stroke-width="1.5" fill="none"/>`;
      }
    });
    if (r < rows.length - 1) {
      const last = row[row.length - 1];
      body += `<path d="M${last.x + last.w / 2},${y + boxH + 4} v10 H${pad + 36} v10 m-4,-5 l4,5 l4,-5" stroke="${C.sub}" stroke-width="1.5" fill="none"/>`;
    }
  });
  let H = h.y + rows.length * (boxH + 34) - 20;
  if (spec.back) {
    const a = pos[spec.back.from], b = pos[spec.back.to];
    const y = Math.max(a.y, b.y) + boxH;
    const ax = a.x + a.w / 2, bx = b.x + b.w / 2;
    body += `<path d="M${ax},${y + 4} v16 H${bx} V${b.y + boxH + 6} m-4,5 l4,-5 l4,5" stroke="${C.main}" stroke-width="1.5" fill="none" stroke-dasharray="4 3"/>`;
    body += `<text x="${(ax + bx) / 2}" y="${y + 36}" font-size="12" text-anchor="middle" fill="${C.main}">${esc(spec.back.label || '')}</text>`;
    H = Math.max(H, y + 46);
  }
  return wrap(W, H + 8, body);
}

const [specPath, outPath, flag] = process.argv.slice(2);
if (!specPath || !outPath) { console.log('사용: node make_figure.cjs <spec.json> <out.svg> [--png]'); process.exit(2); }
const spec = JSON.parse(fs.readFileSync(specPath, 'utf8'));
const svg = { bar, change, flow }[spec.type](spec);
fs.writeFileSync(outPath, svg);
if (flag === '--png') {
  const chrome = ['C:/Program Files/Google/Chrome/Application/chrome.exe', 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'].find((p) => fs.existsSync(p));
  const m = svg.match(/width="(\d+)" height="(\d+)"/);
  const png = outPath.replace(/\.svg$/, '.png');
  const html = outPath.replace(/\.svg$/, '.html');
  fs.writeFileSync(html, `<!doctype html><meta charset="utf-8"><body style="margin:0">${svg}</body>`);
  execFileSync(chrome, ['--headless=new', '--disable-gpu', '--hide-scrollbars', '--force-device-scale-factor=2', `--window-size=${m[1]},${m[2]}`, `--user-data-dir=${path.join(path.dirname(path.resolve(outPath)), '.chrome-prof')}`, `--screenshot=${path.resolve(png)}`, 'file:///' + path.resolve(html).replace(/\\/g, '/')], { stdio: 'ignore' });
  fs.unlinkSync(html);
  console.log(png);
}
