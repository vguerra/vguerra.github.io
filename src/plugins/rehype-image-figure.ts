import type { Element, Root } from "hast";
import { h } from "hastscript";
import { SKIP, visit } from "unist-util-visit";

/**
 * Wraps images that carry a markdown title — ![alt](src "Caption") — in a
 * <figure> with the title as <figcaption>. Runs after rehype-unwrap-images,
 * so it matches bare <img> nodes rather than <p>-wrapped ones.
 */
export function rehypeImageFigure() {
	return (tree: Root) => {
		visit(tree, "element", (node: Element, index, parent) => {
			if (node.tagName !== "img" || !parent || index === undefined) return;
			const title = node.properties.title;
			if (typeof title !== "string" || title.trim() === "") return;
			delete node.properties.title;
			parent.children[index] = h("figure", [node, h("figcaption", title)]);
			return SKIP;
		});
	};
}
