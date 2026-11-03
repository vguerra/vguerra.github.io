import type { SiteConfig } from "@/types";

export const siteConfig: SiteConfig = {
	url: "https://blog.vguerra.net/",
	title: "Victor Guerra",
	author: "Victor Guerra",
	description: "Deep learning frameworks, compilers, and ML frontier research.",
	lang: "en-US",
	ogLocale: "en_US",
	date: {
		locale: "en-US",
		options: {
			day: "numeric",
			month: "short",
			year: "numeric",
		},
	},
};

// Used to generate links in both the Header & Footer.
export const menuLinks: { path: string; title: string }[] = [
	{
		path: "/",
		title: "Home",
	},
	{
		path: "/posts/",
		title: "Blog",
	},
	{
		path: "/notes/",
		title: "Notes",
	},
	{
		path: "/cv/",
		title: "CV",
	},
	{
		path: "/about/",
		title: "About",
	},
];
