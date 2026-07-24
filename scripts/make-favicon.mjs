#!/usr/bin/env node
/**
 * make-favicon — regenerate public/icon.svg from the dragon logo art.
 *
 * Produces a single theme-aware SVG favicon that embeds two raster variants:
 *   - the dark (navy) dragon, shown in light colour schemes
 *   - the light (white) dragon, shown in dark colour schemes  (prefers-color-scheme)
 * Each is cropped to just the dragon (the "VG" wordmark is dropped — illegible at
 * favicon sizes), trimmed, and padded to a square canvas.
 *
 * astro-webmanifest rasterises this icon.svg into the PNG/apple-touch fallbacks
 * at build time, so this is the only favicon source to maintain.
 *
 * Usage:  pnpm make-favicon
 */
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const root = new URL("../", import.meta.url);
const SRC_DARK = fileURLToPath(new URL("public/images/dragon-light.png", root)); // navy dragon
const SRC_LIGHT = fileURLToPath(new URL("public/images/dragon-dark.png", root)); // white dragon
const DEST = fileURLToPath(new URL("public/icon.svg", root));

const CANVAS = 512;
const INNER = 460; // dragon fits within this; remainder is padding
const MARGIN = (CANVAS - INNER) / 2;

/** Crop a portrait dragon+wordmark PNG to just the dragon, padded to a square PNG buffer. */
async function dragonSquare(src) {
	const whole = await sharp(src).trim({ threshold: 10 }).toBuffer();
	const { height, width } = await sharp(whole).metadata();
	// Keep the top ~74% (the dragon); drop the "VG" wordmark below it.
	const dragon = await sharp(whole)
		.extract({ left: 0, top: 0, width, height: Math.round(height * 0.74) })
		.trim({ threshold: 10 })
		.toBuffer();
	return sharp(dragon)
		.resize(INNER, INNER, { fit: "contain", background: { r: 0, g: 0, b: 0, alpha: 0 } })
		.extend({
			top: MARGIN,
			bottom: MARGIN,
			left: MARGIN,
			right: MARGIN,
			background: { r: 0, g: 0, b: 0, alpha: 0 },
		})
		.png({ compressionLevel: 9 })
		.toBuffer();
}

const dark = await dragonSquare(SRC_DARK);
const light = await dragonSquare(SRC_LIGHT);

const svg =
	`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${CANVAS} ${CANVAS}">` +
	"<style>.d{display:none}@media(prefers-color-scheme:dark){.l{display:none}.d{display:inline}}</style>" +
	`<image class="l" href="data:image/png;base64,${dark.toString("base64")}" width="${CANVAS}" height="${CANVAS}"/>` +
	`<image class="d" href="data:image/png;base64,${light.toString("base64")}" width="${CANVAS}" height="${CANVAS}"/>` +
	"</svg>\n";

writeFileSync(DEST, svg);
console.log(`✓ wrote public/icon.svg (${(svg.length / 1024).toFixed(0)}KB, theme-aware dragon)`);
