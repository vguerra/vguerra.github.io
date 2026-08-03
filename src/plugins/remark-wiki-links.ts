import type { Link, Root, Text } from "mdast";
import { SKIP, visit } from "unist-util-visit";

/**
 * Rewrites Obsidian-style wiki links in note bodies into real links:
 *   [[regression-metrics]]        -> <a href="/notes/regression-metrics/">regression-metrics</a>
 *   [[regression-metrics|OLS]]    -> <a href="/notes/regression-metrics/">OLS</a>
 *
 * The slug is used verbatim as the note id (matches the source wiki filenames),
 * so links resolve to /notes/<slug>/. Runs on mdast text nodes; math and code
 * nodes are separate node types and are left untouched.
 */
const WIKI_LINK = /\[\[([^\]|]+?)(?:\|([^\]]+?))?\]\]/g;

export function remarkWikiLinks() {
	return (tree: Root) => {
		visit(tree, "text", (node: Text, index, parent) => {
			if (!parent || index === undefined || !node.value.includes("[[")) return;

			const value = node.value;
			const replacement: Array<Text | Link> = [];
			let cursor = 0;
			WIKI_LINK.lastIndex = 0;
			let match: RegExpExecArray | null = WIKI_LINK.exec(value);
			while (match !== null) {
				const [full, slugRaw, labelRaw] = match;
				if (match.index > cursor) {
					replacement.push({ type: "text", value: value.slice(cursor, match.index) });
				}
				if (slugRaw !== undefined) {
					replacement.push({
						type: "link",
						url: `/notes/${slugRaw.trim()}/`,
						children: [{ type: "text", value: (labelRaw ?? slugRaw).trim() }],
					});
				}
				cursor = match.index + full.length;
				match = WIKI_LINK.exec(value);
			}

			if (!replacement.length) return;
			if (cursor < value.length) {
				replacement.push({ type: "text", value: value.slice(cursor) });
			}

			parent.children.splice(index, 1, ...replacement);
			return [SKIP, index + replacement.length];
		});
	};
}
