const STOPWORDS = new Set(["a", "an", "the", "on", "of", "to", "in", "and", "for", "how", "why"]);

/** "Victor Guerra" -> "Guerra, Victor" (BibTeX's "Last, First" form). */
export function bibtexAuthor(name: string): string {
	const parts = name.trim().split(/\s+/);
	if (parts.length < 2) return name.trim();
	const last = parts.pop() as string;
	return `${last}, ${parts.join(" ")}`;
}

/** Escape characters that are special to BibTeX/LaTeX inside a braced field. */
function escapeField(value: string): string {
	return value.replace(/[&%#$_]/g, (c) => `\\${c}`);
}

/** Citation key: <handle><year><first meaningful title word>, e.g. vguerra2026bootstrapping. */
export function bibtexKey(handle: string, year: number, title: string): string {
	const word =
		title
			.toLowerCase()
			.split(/[^a-z0-9]+/)
			.find((w) => w && !STOPWORDS.has(w)) ?? "post";
	return `${handle}${year}${word}`;
}

interface BibtexInput {
	author: string;
	handle: string;
	title: string;
	date: Date;
	url: string;
}

const MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];

/** A @misc entry for a blog post. The title is double-braced so BibTeX keeps its capitalisation. */
export function bibtexEntry({ author, handle, title, date, url }: BibtexInput): string {
	const year = date.getUTCFullYear();
	return [
		`@misc{${bibtexKey(handle, year, title)},`,
		`  author = {${escapeField(bibtexAuthor(author))}},`,
		`  title  = {{${escapeField(title)}}},`,
		`  year   = {${year}},`,
		`  month  = ${MONTHS[date.getUTCMonth()]},`,
		`  url    = {${url}}`,
		"}",
	].join("\n");
}
