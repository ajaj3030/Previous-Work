import ReactFlow, {
	MiniMap,
	Controls,
	Background,
	useNodesState,
	useEdgesState,
	addEdge,
	Handle,
	MarkerType,
	useReactFlow,
	useStoreApi,
	Position,
} from "reactflow";

export const CustomNode = ({ data }) => {
	return (
		<div className="flex items-center justify-center w-6 h-6 bg-blue-200 rounded-full text-purple text-xs">
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
		</div>
	);
};
