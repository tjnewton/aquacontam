/**
 * Export the contamination flow field banner as PNG using puppeteer.
 * Run via: npx -p puppeteer node assets/export-header.cjs
 */
const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const htmlPath = path.resolve(__dirname, 'header-generator.html');
const outputPath = path.resolve(__dirname, 'header.png');

(async () => {
    console.log('Launching browser...');
    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    const page = await browser.newPage();
    await page.setViewport({ width: 1600, height: 800 });

    console.log('Loading HTML...');
    await page.goto('file://' + htmlPath, { waitUntil: 'networkidle0', timeout: 30000 });

    console.log('Waiting for animation to complete (~500 frames)...');
    await page.waitForFunction('window.animationDone === true', { timeout: 120000 });

    console.log('Extracting canvas as PNG...');
    const dataUrl = await page.evaluate(() => {
        const canvas = document.querySelector('#canvas-container canvas');
        return canvas.toDataURL('image/png');
    });

    const base64 = dataUrl.replace(/^data:image\/png;base64,/, '');
    fs.writeFileSync(outputPath, Buffer.from(base64, 'base64'));

    const size = fs.statSync(outputPath).size;
    console.log('Saved to ' + outputPath);
    console.log('File size: ' + (size / 1024).toFixed(1) + ' KB');

    await browser.close();
})();
