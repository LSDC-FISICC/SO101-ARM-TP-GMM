// Render docs/act_report.html to PDF (A4, light theme, print layout): NODE_PATH=$(npm root -g) node scripts/act/report_to_pdf.js docs/act_report.html docs/act_report.pdf
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const body = fs.readFileSync(process.argv[2], 'utf8');
  const html = `<!doctype html><html lang="es" data-theme="light"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{color-scheme:light}
body{margin:0;font-size:13.5px !important;line-height:1.5 !important}
@page{size:A4;margin:14mm 12mm}
.wrap{padding-block:0 !important;max-width:none !important}
figure,.verdict,.scroll,tr,li{break-inside:avoid}
h2,h3{break-after:avoid}
.wrap{gap:18px !important}
table{min-width:0 !important;font-size:11px !important}
th,td{padding:4px 6px !important;white-space:normal !important}
thead th{font-size:9.5px !important}
pre{white-space:pre-wrap !important;word-break:break-all;font-size:10.5px !important}
.charts{grid-template-columns:1fr 1fr !important}
</style></head><body>${body}</body></html>`;
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.emulateMedia({ colorScheme: 'light', media: 'print' });
  await page.setContent(html, { waitUntil: 'networkidle', timeout: 60000 }).catch(() => {});
  await page.waitForTimeout(800);
  await page.pdf({ path: process.argv[3], format: 'A4', printBackground: true,
    margin: { top: '14mm', bottom: '16mm', left: '12mm', right: '12mm' },
    displayHeaderFooter: true, headerTemplate: '<span></span>',
    footerTemplate: '<div style="font-size:8px;color:#888;width:100%;text-align:center;font-family:sans-serif">ACT en dataset3: A vs B · <span class="pageNumber"></span>/<span class="totalPages"></span></div>' });
  await browser.close();
})();
