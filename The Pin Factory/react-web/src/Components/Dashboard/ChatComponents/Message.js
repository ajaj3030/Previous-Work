import React, { useState, useRef, useEffect } from "react";
import AvatarProfilePicture from "../../../assets/baby-animation/profile_picture.png";
import ProfilePicture from "../../../assets/test_pp/IMG_2109.PNG";

import clsx from "clsx";

import "highlight.js/styles/agate.css";
import hljs from "highlight.js";

export default function Message({ task, index }) {
	
	const [showOriginalPrompt, setShowOriginalPrompt] = useState(false);
	const taskRef = useRef(); // Reference to the task element
	const handleOriginalPromptClick = (event) => {
		event.stopPropagation(); // Prevent the click from being captured by the outside click listener
		setShowOriginalPrompt(!showOriginalPrompt);
	};

	// Outside click listener
	useEffect(() => {
		const handleClickOutside = (event) => {
			if (taskRef.current && !taskRef.current.contains(event.target)) {
				setShowOriginalPrompt(false); // Hide the original prompt when a click occurs outside the task element
			}
		};

		document.addEventListener("mousedown", handleClickOutside);
		return () => {
			document.removeEventListener("mousedown", handleClickOutside);
		};
	}, []);

	return (
		<div
			className={`flex flex-row w-full items-center justify-center py-3 focus:outline-none group ${
				index % 2 === 0 ? "bg-gray-50" : "bg-white"
			}`}>
			<div className="flex flex-row w-2/3 h-full justify-start items-start gap-x-6 items-start py-1">
				{/* SENDER PROFILE PICTURE */}
				{task.sender === "George Duffy" ? (
					<div className="w-10 h-10 relative flex-shrink-0 rounded-full outline outline-2 outline-gray-900/10">
						<img
							src={ProfilePicture}
							alt="human name"
							className="absolute top-0 left-0 w-full h-full rounded-full object-cover"
						/>
					</div>
				) : (
					<div className="flex flex-row items-center justify-center w-10 h-10 pt-1 px-1 bg-orange-400 rounded-full outline outline-2 outline-gray-900/10 overflow-hidden flex-shrink-0">
						<img
							src={AvatarProfilePicture}
							className="w-full h-full object-cover"
						/>
					</div>
				)}

				{/* TOP 2 OF MAIN CONTENT */}
				<MessageBody text={task.content} />
			</div>
		</div>
	);
}

const languageNames = {
	js: "JavaScript",
	ts: "TypeScript",
	javascript: "JavaScript",
	typescript: "TypeScript",
	php: "PHP",
	python: "Python",
	ruby: "Ruby",
	go: "Go",
	bash: "cURL",
};

export function MessageBody({ text }) {

	const practice = "Hello there, here is some **bold text** and here is some *italic text* and here is some ***bold and italic text***. Here is some `inline code` and here is some ```\nblock code\n``` and here is some ```js\nconst message = 'Hello World!'\n```. Here is an image: ![alt text](https://firebasestorage.googleapis.com/v0/b/thepinfactory-42d2a.appspot.com/o/dalle2_assets%2Ftest-image.png?alt=media&token=c52a3602-daec-4813-b871-a854c0756fbe) and here is a link: [link](https://www.google.com)."

	const messageSegments = extractCodeAndText(text);

	useEffect(() => {
		console.log("Detecting");
		console.log(text);
		console.log(messageSegments);
		hljs.highlightAll();
	}, []);

	const [imageHovered, setImageHovered] = useState(false);

	return (
		<div className="flex flex-col w-full space-y-1">
			{messageSegments.map((segment, index) => {
				if (segment.type === "code") {
					return (
						<pre key={index}>
							<div className="rounded-lg bg-zinc-800">
								<div className="flex items-start font-sm text-white justify-between px-4 pt-1 border-b border-white/5">
									<h1 className="text-xs font-semibold text-emerald-400 font-sans px-1 pb-3 pt-2 border-b border-emerald-400">
										{languageNames[segment.language]}
									</h1>
									<div className="pt-1">
										<CopyButton code={segment.content} />
									</div>
								</div>
								<div className="py-4 px-5 overflow-y-auto bg-zinc-900 rounded-b-lg">
									<code
										className={`hljs language-${segment.language} text-sm leading-normal`}
										style={{
											background: "none",
											padding: "0px",
										}}>
										{segment.content}
									</code>
								</div>
							</div>
						</pre>
					);
				}
				if (segment.type === "text") {
					return <p key={index}>{segment.content}</p>;
				}
				if (segment.type === "image") {
					return (
						<div className="flex justify-center items-center w-full">
							<div
								key={index}
								className="relative  min-w-sm max-w-sm"
								onMouseEnter={() => setImageHovered(true)}
								onMouseLeave={() => setImageHovered(false)}>
								<img
									src={segment.src}
									alt={segment.alt}
									className="relative w-full h-auto rounded-md"
								/>
								<div
									className={`absolute top-0 right-0 flex justify-center items-center mr-2 mt-2 transition ease-in-out duration 300 ${
										imageHovered
											? "opacity-100"
											: "opacity-0 "
									}`}>
									<SaveButton image={segment.content} />
								</div>
							</div>
						</div>
					);
				}
			})}
		</div>
	);
}
  
  

function extractCodeAndText_OLD(inputStr) {
	if (!inputStr) return [];

	let parts = inputStr.split("```");
	let extractedBlocks = [];

	for (let i = 0; i < parts.length; i++) {
		let trimmedPart = parts[i].trim();

		// Check if the part is empty (could happen if there are multiple backticks in a row)
		if (trimmedPart === "") continue;

		let match = trimmedPart.match(/^(\w+)\n([\s\S]*)/);

		if (match) {
			let language = match[1];
			let content = match[2];

			if (language === "image") {
				try {
					let imageData = JSON.parse(content);
					extractedBlocks.push({
						type: "image",
						content: imageData.url,
						ref: imageData.ref,
						language: null,
					});
				} catch (error) {
					console.error("Unable to parse image data:", error);
				}
			} else {
				extractedBlocks.push({
					type: "code",
					content: content,
					language: language,
				});
			}
		} else {
			let content = trimmedPart;
			let contentLines = content.split("\n");

			content = contentLines.map((line, lineIndex) => {
				let chunks = line.split(/(\[[^\]]+\]\([^\)]+\))/);
				let parts = chunks.map((chunk, partIndex) => {
					let linkMatch = chunk.match(/^\[([^\]]+)\]\(([^\)]+)\)$/);
					if (linkMatch) {
						return (
							<a
								href={linkMatch[2]}
								key={partIndex}
								className="font-semibold hover:opacity-70 underline-offset-2 text-purple-600"
								target="_blank">
								{linkMatch[1]}
							</a>
						);
					} else {
						let boldedChunks = chunk.split(/`([^`]*)`/g);
						let boldedParts = boldedChunks.map(
							(boldChunk, boldPartIndex) => {
								return boldPartIndex % 2 === 1 ? (
									<b key={boldPartIndex}>{boldChunk}</b>
								) : (
									boldChunk
								);
							},
						);
						return boldedParts;
					}
				});

				return (
					<React.Fragment key={lineIndex}>
						{parts}
						{lineIndex < contentLines.length - 1 && <br />}
					</React.Fragment>
				);
			});

			extractedBlocks.push({
				type: "text",
				content: content,
				language: null,
			});
		}
	}
	return extractedBlocks;
}


function extractCodeAndText(inputStr) {
	if (!inputStr) return [];
  
	let parts = inputStr.split("```");
	let extractedBlocks = [];
  
	for (let i = 0; i < parts.length; i++) {
	  let trimmedPart = parts[i].trim();
  
	  if (trimmedPart === "") continue;
  
	  let match = trimmedPart.match(/^(\w+)\n([\s\S]*)/);
  
	  if (match) {
		let language = match[1];
		let content = match[2];
  
		if (language !== "image") {
		  extractedBlocks.push({
			type: "code",
			content: content,
			language: language,
		  });
		}
	  } else {
		let content = trimmedPart;
		let contentLines = content.split("\n");
  
		content = contentLines.map((line, lineIndex) => {
		  let chunks = line.split(/(\[[^\]]+\]\([^\)]+\)|!\[[^\]]+\]\([^\)]+\))/);
		  let parts = chunks.map((chunk, partIndex) => {
			let imageMatch = chunk.match(/^!\[([^\]]+)\]\(([^\)]+)\)$/);
			let linkMatch = chunk.match(/^\[([^\]]+)\]\(([^\)]+)\)$/);
			if (imageMatch) {
				extractedBlocks.push({
					type: "image",
					alt: imageMatch[1],
					src: imageMatch[2],
				});
			} else if (linkMatch) {
			  return (
				<a
				  href={linkMatch[2]}
				  key={partIndex}
				  className="font-semibold hover:opacity-70 underline-offset-2 text-purple-600"
				  target="_blank"
				  rel="noreferrer"
				>
				  {linkMatch[1]}
				</a>
			  );
			} else {
				
				let boldedAndItalicisedChunks = chunk.split(/\*{3}/g);

				let boldedAndItalicisedParts = boldedAndItalicisedChunks.map((boldAndItalicisedChunk, boldPartIndex) => {
				if (boldPartIndex % 2 === 1) {
					return <b key={boldPartIndex}><i>{boldAndItalicisedChunk}</i></b>;
				} else {
					let boldChunks = boldAndItalicisedChunk.split(/\*{2}/g);
					return boldChunks.map((boldChunk, boldIndex) => {
					if (boldIndex % 2 === 1) {
						return <b key={boldPartIndex + "-" + boldIndex}>{boldChunk}</b>;
					} else {
						let italicChunks = boldChunk.split(/\*{1}/g);
						return italicChunks.map((italicChunk, italicIndex) => {
						if (italicIndex % 2 === 1) {
							return <i key={boldPartIndex + "-" + boldIndex + "-" + italicIndex}>{italicChunk}</i>;
						} else {
							let codeChunks = italicChunk.split(/`{1}/g);
							return codeChunks.map((codeChunk, codeIndex) => {
							return codeIndex % 2 === 1 ? <span className="px-1"><span className="bg-zinc-800/80 px-2 py-1 text-white rounded-md font-mono font-semibold text-xs leading-normal" key={boldPartIndex + "-" + boldIndex + "-" + italicIndex + "-" + codeIndex}>{codeChunk}</span></span> : codeChunk;
							});
						}
						}).flat();
					}
					}).flat();
				}
				});
				return boldedAndItalicisedParts.flat();
			}
		  });
  
		  const text_content = (
			<React.Fragment key={lineIndex}>
			  {parts}
			  {lineIndex < contentLines.length - 1 && <br />}
			</React.Fragment>
		  );

		  extractedBlocks.push({
			type: "text",
			content: text_content,
		  });

		});
	  }
	}
	return extractedBlocks;
}
  
  
  
	  
  

  
function ClipboardIcon(props) {
	return (
		<svg viewBox="0 0 20 20" aria-hidden="true" {...props}>
			<path
				strokeWidth="0"
				d="M5.5 13.5v-5a2 2 0 0 1 2-2l.447-.894A2 2 0 0 1 9.737 4.5h.527a2 2 0 0 1 1.789 1.106l.447.894a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-5a2 2 0 0 1-2-2Z"
			/>
			<path
				fill="none"
				strokeLinejoin="round"
				d="M12.5 6.5a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-5a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2m5 0-.447-.894a2 2 0 0 0-1.79-1.106h-.527a2 2 0 0 0-1.789 1.106L7.5 6.5m5 0-1 1h-3l-1-1"
			/>
		</svg>
	);
}

function DownloadIcon(props) {
	return (
		<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" {...props}>
			<path
				stroke-linecap="round"
				stroke-linejoin="round"
				d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3"
			/>
		</svg>
	);
}

function CopyButton({ code }) {
	let [copyCount, setCopyCount] = useState(0);
	let copied = copyCount > 0;

	useEffect(() => {
		if (copyCount > 0) {
			let timeout = setTimeout(() => setCopyCount(0), 1000);
			return () => {
				clearTimeout(timeout);
			};
		}
	}, [copyCount]);

	return (
		<button
			type="button"
			className={clsx(
				"relative group/button overflow-hidden rounded-full py-1 pl-2 pr-3 text-2xs font-medium opacity-100 transition focus:opacity-100",
				copied
					? "bg-emerald-400/10 ring-1 ring-inset ring-emerald-400/20"
					: "",
			)}
			onClick={() => {
				window.navigator.clipboard.writeText(code).then(() => {
					setCopyCount((count) => count + 1);
				});
			}}>
			<span
				aria-hidden={copied}
				className={clsx(
					"text-xs font-bold pointer-events-none flex items-center gap-0.5 text-zinc-400 transition duration-300",
					copied && "-translate-y-1.5 opacity-0",
				)}>
				<ClipboardIcon className="h-5 w-5 fill-zinc-500/20 stroke-zinc-500 transition-colors group-hover/button:stroke-zinc-400" />
				Copy
			</span>
			<span
				aria-hidden={!copied}
				className={clsx(
					"text-xs font-bold pointer-events-none absolute inset-0 flex items-center justify-center text-emerald-400 transition duration-300",
					!copied && "translate-y-1.5 opacity-0",
				)}>
				Copied!
			</span>
		</button>
	);
}

function SaveButton({ image }) {
	let [saved, setSaved] = useState(0);

	const handleDownload = async () => {
		setSaved(1);
		const response = await fetch(image);
		const blob = await response.blob();
		const url = URL.createObjectURL(blob);
		const link = document.createElement("a");
		link.href = url;
		link.download = "test.jpg";
		document.body.appendChild(link);
		link.click();

		document.body.removeChild(link);
		URL.revokeObjectURL(url);

		let timeout = setTimeout(() => setSaved(0), 2000);
		return () => {
			clearTimeout(timeout);
		};
	};

	const ref = useRef(null);
	const [width, setWidth] = useState(0);
	const [height, setHeight] = useState(0);

	useEffect(() => {
		setWidth(ref.current.offsetWidth);
		setHeight(ref.current.offsetHeight);
	}, []);

	return (
		<button
			type="button"
			className={`
        relative group/button rounded-md bg-zinc-100 py-1.5 text-2xs font-medium opacity-100 transition ease-in-out duration-300 focus:opacity-100
      ${saved === 0 ? "px-2.5" : "px-7"}`}
			onClick={handleDownload}>
			<span
				ref={ref}
				className={clsx(
					"text-xs font-semibold pointer-events-none absolute relative flex items-center gap-0.5 text-zinc-800 group-hover/button:stroke-zinc-600 transition duration-300",
					saved > 0 && "-translate-y-1.5 opacity-0",
				)}>
				<DownloadIcon className="h-4 w-4 fill-zinc-500/20 stroke-zinc-800 stroke-[1.6px] transition-colors group-hover/button:stroke-zinc-500" />
				Save
			</span>
			<span
				className={clsx(
					"text-xs font-semibold pointer-events-none absolute inset-0 flex items-center justify-center text-zinc-800 transition duration-300",
					saved === 0 && "translate-y-1.5 opacity-0",
				)}>
				<svg
					className="animate-spin mr-2 h-4 w-4 text-zinc-800 inline-block"
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24">
					<circle
						className="opacity-25"
						cx="12"
						cy="12"
						r="10"
						stroke="currentColor"
						strokeWidth="4"></circle>
					<path
						className="opacity-75"
						fill="currentColor"
						d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
				</svg>
				Processing
			</span>
		</button>
	);
}
