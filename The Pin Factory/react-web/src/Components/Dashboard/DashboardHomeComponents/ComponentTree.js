import ReactFlow, {
	MiniMap,
	Controls,
	Background,
	Handle,
	useReactFlow,
} from "reactflow";
import React, { useState, useEffect } from "react";
import { hierarchy, tree } from "d3-hierarchy";

import { functions, httpsCallable, auth } from "../../../firebase";

// this is the original component tree. the tree used at the bottom does not have props in it, and is used for easy development
export const ComponentTree = ({ thread }) => {
	// Store them in states
	const [nodes, setNodes] = useState([]);
	const [edges, setEdges] = useState([]);
	const [layoutedNodes, setLayoutedNodes] = useState([]);

	const [allThreadMessages, setallThreadMessages] = useState([]);
	const getthreadmessages = httpsCallable(functions, "getthreadmessages");
	useEffect(() => {
		getthreadmessages({ threadId: thread.threadId })
			.then((result) => {
				console.log("this is the data", result.data.data);
				setallThreadMessages(result.data.data);
			})
			.catch((error) => {
				// Handle any errors here.
				console.error(error);
			});
	}, []); // empty dependency array to run once

	useEffect(() => {
		// Make sure allThreadMessages is not empty
		if (allThreadMessages.length > 0) {
			// Generate nodes and edges from allThreadMessages
			const newNodes = allThreadMessages.map((message) => ({
				id: message.messageId,
				type: "custom",
				parentId: message.parentId,
				data: { label: message.content.substring(0, 3) }, // You can customize what information to display on each node
				// Position will be set later by the generateTreeLayout function
				position: { x: 0, y: 0 },
			}));

			const newEdges = allThreadMessages
				// Filter out messages that don't have a parent
				.filter((message) => message.parentId)
				// Map each message to an edge
				.map((message) => ({
					id: `e${message.parentId}-${message.messageId}`,
					source: message.parentId,
					target: message.messageId,
					...edgeOptions,
				}));

			// Set nodes and edges states
			setNodes(newNodes);
			setEdges(newEdges);
		}
	}, [allThreadMessages]);
	useEffect(() => {
		if (nodes.length > 0 && edges.length > 0) {
			const newLayoutedNodes = generateTreeLayout(nodes, edges);
			setLayoutedNodes(newLayoutedNodes);
		}
	}, [nodes, edges]);

	const edgeOptions = {
		type: "step",
		animated: true,
		style: {
			// use 0.2s to speed up the animation
			animation: "dashdraw 0.8s linear infinite",
			strokeDasharray: 5,
			strokeLinejoin: "round",
			stroke: "blue",
			strokeWidth: 2,
			strokeOpacity: 0.4,
		},
	};

	const CustomNode = ({ data }) => {
		const [showTooltip, setShowTooltip] = useState(false);

		const handleMouseEnter = () => {
			setShowTooltip(true);
		};

		const handleMouseLeave = () => {
			setShowTooltip(false);
		};
		return (
			<div
				className="flex items-center justify-center w-6 h-6 bg-blue-200 rounded-full text-purple text-xs"
				onMouseEnter={handleMouseEnter}
				onMouseLeave={handleMouseLeave}
				style={data.style}>
				<Handle
					type="target"
					position="top"
					style={{ borderRadius: "50%" }}
				/>
				{data.label}
				<Handle
					type="source"
					position="bottom"
					style={{ borderRadius: "50%" }}
				/>
				{showTooltip && (
					<div className="absolute top-1/2 left-full bg-white p-1 rounded-md shadow-md z-50 text-black">
						Tooltip Content
					</div>
				)}
			</div>
		);
	};

	const generateTreeLayout = (nodes, edges) => {
		// Create a mapping of node IDs to node objects
		const nodeMap = new Map(nodes.map((node) => [node.id, node]));

		// Create a tree structure
		const rootId = nodes.find((node) => node.parentId === null).id;
		const root = {
			id: rootId,
			...nodeMap.get(rootId),
			children: [],
		};

		const stack = [root];
		while (stack.length) {
			const node = stack.pop();

			// Find edges where this node is the source
			const nodeEdges = edges.filter((edge) => edge.source === node.id);

			// Add the target nodes of these edges as children of this node
			node.children = nodeEdges.map((edge) => {
				const childNode = {
					id: edge.target,
					...nodeMap.get(edge.target),
					children: [],
				};
				stack.push(childNode);
				return childNode;
			});
		}

		// Create a d3 hierarchy from the root
		const d3Root = hierarchy(root);

		// Create a tree layout function
		const treeLayout = tree()
			.size([800, 600]) // You can adjust this to fit your component size
			.separation((a, b) => (a.parent === b.parent ? 1 : 2)); // Adjust this for the desired separation between nodes

		// Calculate the positions of the nodes
		treeLayout(d3Root);

		// Update the positions of the nodes
		const updatedNodes = nodes.map((node) => {
			// Find the corresponding d3 node
			const d3Node = d3Root
				.descendants()
				.find((d) => d.data.id === node.id);

			// If the d3 node was found, update the position of the node
			if (d3Node) {
				return {
					...node,
					position: {
						x: d3Node.x,
						y: d3Node.y,
					},
				};
			}

			// If the d3 node wasn't found, return the node as is
			return node;
		});

		return updatedNodes;
	};
	const nodeTypes = React.useMemo(
		() => ({
			custom: CustomNode,
		}),
		[],
	);
	const [nodeCounter, setNodeCounter] = useState(8); // Initialize nodeCounter state to keep track of total nodes

	const addChildNode = () => {
		if (!nodeClicked) {
			console.error("No node selected.");
			return;
		}
		// Create a new node
		const newNode = {
			id: nodeCounter.toString(),
			type: "custom",
			data: { label: nodeCounter.toString() },
		};

		// Create a new edge from selected node to new node
		const newEdge = {
			id: `e${nodeClicked}-${nodeCounter}`,
			source: nodeClicked,
			target: newNode.id,
			...edgeOptions,
		};

		// Add new node and edge to nodes and edges arrays
		const newNodes = [...nodes, newNode];
		const newEdges = [...edges, newEdge];
		setNodes(newNodes);
		setEdges(newEdges);

		// Generate a tree layout for the new nodes
		const layoutNodes = generateTreeLayout(newNodes, newEdges);
		setNodes(layoutNodes);

		// Increase the node counter
		setNodeCounter(nodeCounter + 1);
	};
	const getPathToNode = (nodeId, edges) => {
		const edge = edges.find((edge) => edge.target === nodeId);
		if (!edge) {
			return [nodeId];
		}
		return [...getPathToNode(edge.source, edges), nodeId, edge.id];
	};

	const [highlightedPath, setHighlightedPath] = useState([]);

	useEffect(() => {
		const processedNodes = nodes.map((node) => ({
			...node,
			data: {
				...node.data,
				style: {
					backgroundColor: highlightedPath.includes(node.id)
						? "red"
						: "blue",
					color: "white",
				},
			},
		}));
		setNodes(processedNodes);

		const processedEdges = edges.map((edge) => ({
			...edge,
			style: {
				...edge.style,
				stroke: highlightedPath.includes(edge.id) ? "red" : "blue",
				strokeWidth: highlightedPath.includes(edge.id) ? 4 : 2,
			},
		}));

		setEdges(processedEdges);
	}, [highlightedPath]);
	const reactFlowInstance = useReactFlow();

	const { getEdges } = useReactFlow();

	const onNodeClick = (event, node) => {
		const edges = getEdges();
		setHighlightedPath(getPathToNode(node.id, edges));
		setNodeClicked(node.id);
	};

	const onPaneClick = () => setHighlightedPath([]);

	const [nodeClicked, setNodeClicked] = useState(false);

	const findChildrenNodes = (nodeId, edges) => {
		const childrenEdges = edges.filter((edge) => edge.source === nodeId);
		return childrenEdges.flatMap((edge) => [
			edge.target,
			...findChildrenNodes(edge.target, edges),
		]);
	};

	return (
		<div className="h-screen w-1/2">
			<div className="flex flex-row">
				<button
					onClick={() => {
						addChildNode();
					}}
					className="w-screen h-10">
					Add child to {` NODE ` + nodeClicked}{" "}
				</button>
			</div>

			{layoutedNodes.length > 0 && (
				<ReactFlow
					nodes={layoutedNodes}
					edges={edges}
					nodeTypes={nodeTypes}
					onNodeClick={onNodeClick}
					onPaneClick={onPaneClick}>
					<MiniMap
						nodeColor={(node) =>
							highlightedPath.includes(node.id) ? "red" : "blue"
						}
					/>
					<Controls />
					<Background variant="dots" gap={25} size={1} />
				</ReactFlow>
			)}
		</div>
	);
};

export default ComponentTree;
