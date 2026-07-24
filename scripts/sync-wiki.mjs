#!/usr/bin/env node
/**
 * sync-wiki — import study notes from an external wiki directory into the blog.
 *
 * The wiki's README.md is the source of truth for organisation: its `##` section
 * headers become categories, and each `- [file.md](file.md) — summary` line
 * provides a note's category, ordering, and description. Each note's first `#`
 * heading becomes its title; the file's mtime becomes its "updated" date.
 *
 * Output (frontmatter + body) is written to src/content/note/ and committed, so
 * the blog builds without needing the wiki present (e.g. in CI).
 *
 * Usage:  pnpm sync-wiki           (uses the default path below)
 *         WIKI_SRC=/path pnpm sync-wiki
 */
import {
	existsSync,
	mkdirSync,
	readdirSync,
	readFileSync,
	rmSync,
	statSync,
	writeFileSync,
} from "node:fs";
import { basename, join } from "node:path";
import { fileURLToPath } from "node:url";

const WIKI_SRC = process.env.WIKI_SRC || "/usr/local/dev/ml/interview-prep/wiki";
const DEST = fileURLToPath(new URL("../src/content/note/", import.meta.url));

/** Parse README.md into a map: filename -> { category, description, order }. */
function parseReadme(src) {
	const readme = readFileSync(join(src, "README.md"), "utf8");
	const meta = new Map();
	let category = "Uncategorized";
	let order = 0;
	const linkRe = /^\s*-\s*\[[^\]]+\]\(([^)]+)\)\s*(?:[—–-]\s*(.*))?$/;
	for (const line of readme.split("\n")) {
		const header = line.match(/^##\s+(.*)$/);
		if (header) {
			category = header[1].trim();
			continue;
		}
		const link = line.match(linkRe);
		if (link) {
			const file = basename(link[1].trim());
			const description = (link[2] || "").trim();
			meta.set(file, { category, description, order: order++ });
		}
	}
	return meta;
}

/** YAML-safe scalar via JSON double-quoting. */
function yaml(value) {
	return JSON.stringify(value);
}

function main() {
	if (!existsSync(WIKI_SRC)) {
		console.error(`✗ Wiki source not found: ${WIKI_SRC}`);
		console.error("  Set WIKI_SRC=/path/to/wiki and retry.");
		process.exit(1);
	}

	const meta = parseReadme(WIKI_SRC);

	// Rebuild the destination from scratch so deletions in the wiki propagate.
	rmSync(DEST, { recursive: true, force: true });
	mkdirSync(DEST, { recursive: true });

	const files = readdirSync(WIKI_SRC)
		.filter((f) => f.endsWith(".md") && f !== "README.md")
		.sort();

	let count = 0;
	let uncategorized = 0;
	let fallbackOrder = 1000;

	for (const file of files) {
		const raw = readFileSync(join(WIKI_SRC, file), "utf8");
		const lines = raw.split("\n");

		// First `# H1` becomes the title; strip it (the page layout renders the title).
		let title = basename(file, ".md");
		let bodyStart = 0;
		for (let i = 0; i < lines.length; i++) {
			const h1 = lines[i].match(/^#\s+(.*)$/);
			if (h1) {
				title = h1[1].trim();
				bodyStart = i + 1;
				break;
			}
		}
		while (bodyStart < lines.length && lines[bodyStart].trim() === "") bodyStart++;
		const body = `${lines.slice(bodyStart).join("\n").trimEnd()}\n`;

		const info = meta.get(file) ?? {
			category: "Uncategorized",
			description: "",
			order: fallbackOrder++,
		};
		if (!meta.has(file)) uncategorized++;

		const updatedDate = statSync(join(WIKI_SRC, file)).mtime.toISOString();

		const frontmatter = [
			"---",
			`title: ${yaml(title)}`,
			`description: ${yaml(info.description)}`,
			`category: ${yaml(info.category)}`,
			`order: ${info.order}`,
			`updatedDate: ${yaml(updatedDate)}`,
			"---",
			"",
		].join("\n");

		writeFileSync(join(DEST, file), frontmatter + body);
		count++;
	}

	console.log(`✓ ${count} notes imported from ${WIKI_SRC}`);
	if (uncategorized) {
		console.log(`  ${uncategorized} not listed in README → filed under "Uncategorized"`);
	}
	console.log("✓ frontmatter generated");
	console.log("✓ wrote to src/content/note/");
}

main();
