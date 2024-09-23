/** @type {import('tailwindcss').Config} */
module.exports = {
	content: ["./src/**/*.{js,jsx,ts,tsx}"],
	theme: {
		extend: {
			dropShadow: {
				"3xl": "0 35px 35px rgba(0, 0, 0, 0.25)",
				"4xl": [
					"0 35px 35px rgba(0, 0, 0, 0.25)",
					"0 45px 65px rgba(0, 0, 0, 0.15)",
				],
			},
			fontFamily: {
				pixelated: ["PixelFontBold", "sans-serif"],
			},
			colors: {
				transparent: "transparent",
				current: "currentColor",
				dark: {
					100: "#444654",
					200: "#343541",
				},
				brand: {
					sd: "#15803d",
					hover: "#16a34a",
				}
			},
		},
	},
	plugins: [
		require("@tailwindcss/line-clamp"),
		// require("@tailwindcss/forms"),
	],
};
