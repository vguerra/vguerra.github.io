import { getCollection } from "astro:content";
import rss from "@astrojs/rss";
import { siteConfig } from "@/site.config";

export const GET = async () => {
	const notes = (await getCollection("note")).sort((a, b) => a.data.order - b.data.order);

	return rss({
		title: `${siteConfig.title} — Notes`,
		description: "Study notes on machine learning, deep learning, and ML systems.",
		site: import.meta.env.SITE,
		items: notes.map((note) => ({
			title: note.data.title,
			description: note.data.description,
			pubDate: note.data.updatedDate ?? new Date(),
			link: `notes/${note.id}/`,
		})),
	});
};
