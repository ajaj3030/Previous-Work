// XMark Icon
const XMark = ({ size = '6', color = 'currentColor', strokeWidth = '0.75' }) => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill={color} className={`w-${size} h-${size}`}>
        <path stroke={color} strokeWidth={strokeWidth} fill-rule="evenodd" d="M5.47 5.47a.75.75 0 011.06 0L12 10.94l5.47-5.47a.75.75 0 111.06 1.06L13.06 12l5.47 5.47a.75.75 0 11-1.06 1.06L12 13.06l-5.47 5.47a.75.75 0 01-1.06-1.06L10.94 12 5.47 6.53a.75.75 0 010-1.06z" clip-rule="evenodd" />
    </svg>
);

const Search = ({ size = '6', color = 'currentColor', strokeWidth = '0.75' }) => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill={color} className={`w-${size} h-${size}`}>
        <path stroke={color} strokeWidth={strokeWidth} fill-rule="evenodd" d="M10.5 3.75a6.75 6.75 0 100 13.5 6.75 6.75 0 000-13.5zM2.25 10.5a8.25 8.25 0 1114.59 5.28l4.69 4.69a.75.75 0 11-1.06 1.06l-4.69-4.69A8.25 8.25 0 012.25 10.5z" clip-rule="evenodd" />
    </svg>
);

const PlusIcon = ({ size = '6', color = 'currentColor', strokeWidth = '0.75' }) => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill={color} className={`w-${size} h-${size}`}>
        <path stroke={color} strokeWidth={strokeWidth} fill-rule="evenodd" d="M12 3.75a.75.75 0 01.75.75v6.75h6.75a.75.75 0 010 1.5h-6.75v6.75a.75.75 0 01-1.5 0v-6.75H4.5a.75.75 0 010-1.5h6.75V4.5a.75.75 0 01.75-.75z" clip-rule="evenodd" />
    </svg>
);

function ThreadsIcon({ color = '#000', height = '16', width = '16', strokeWidth='1'}) {
	return (
		<svg aria-hidden="true" height={height} viewBox="0 0 16 16" version="1.1" width={width} data-view-component="true">
			<path fill={color} strokeWidth={strokeWidth} d="M5 5.372v.878c0 .414.336.75.75.75h4.5a.75.75 0 0 0 .75-.75v-.878a2.25 2.25 0 1 1 1.5 0v.878a2.25 2.25 0 0 1-2.25 2.25h-1.5v2.128a2.251 2.251 0 1 1-1.5 0V8.5h-1.5A2.25 2.25 0 0 1 3.5 6.25v-.878a2.25 2.25 0 1 1 1.5 0ZM5 3.25a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0Zm6.75.75a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Zm-3 8.75a.75.75 0 1 0-1.5 0 .75.75 0 0 0 1.5 0Z"></path>
		</svg>
	);
}

function EngineeringIcon({ color = '#000', height = '16', width = '16', strokeWidth='1' }) {
	return (
		<svg aria-hidden="true" height={height} viewBox="0 0 16 16" version="1.1" width={width} data-view-component="true" class="octicon octicon-git-pull-request UnderlineNav-octicon d-none d-sm-inline">
			<path fill={color} strokeWidth={strokeWidth} d="M1.5 3.25a2.25 2.25 0 1 1 3 2.122v5.256a2.251 2.251 0 1 1-1.5 0V5.372A2.25 2.25 0 0 1 1.5 3.25Zm5.677-.177L9.573.677A.25.25 0 0 1 10 .854V2.5h1A2.5 2.5 0 0 1 13.5 5v5.628a2.251 2.251 0 1 1-1.5 0V5a1 1 0 0 0-1-1h-1v1.646a.25.25 0 0 1-.427.177L7.177 3.427a.25.25 0 0 1 0-.354ZM3.75 2.5a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Zm0 9.5a.75.75 0 1 0 0 1.5.75.75 0 0 0 0-1.5Zm8.25.75a.75.75 0 1 0 1.5 0 .75.75 0 0 0-1.5 0Z"></path>
		</svg>
	);
}

function CheckIcon({ color = '#000', height = '16', width = '16', strokeWidth='1' }) {
	return (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" height={height} width={width}>
            <path fill={color} stroke={color} strokeWidth={strokeWidth} d="M19.916 4.626a.75.75 0 01.208 1.04l-9 13.5a.75.75 0 01-1.154.114l-6-6a.75.75 0 011.06-1.06l5.353 5.353 8.493-12.739a.75.75 0 011.04-.208z" />
        </svg>
	);
}


function TrashIcon({ color = '#000', height = '16', width = '16' }) {
	return (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" height={height} width={width}>
            <path fill={color}  d="M16.5 4.478v.227a48.816 48.816 0 013.878.512.75.75 0 11-.256 1.478l-.209-.035-1.005 13.07a3 3 0 01-2.991 2.77H8.084a3 3 0 01-2.991-2.77L4.087 6.66l-.209.035a.75.75 0 01-.256-1.478A48.567 48.567 0 017.5 4.705v-.227c0-1.564 1.213-2.9 2.816-2.951a52.662 52.662 0 013.369 0c1.603.051 2.815 1.387 2.815 2.951zm-6.136-1.452a51.196 51.196 0 013.273 0C14.39 3.05 15 3.684 15 4.478v.113a49.488 49.488 0 00-6 0v-.113c0-.794.609-1.428 1.364-1.452zm-.355 5.945a.75.75 0 10-1.5.058l.347 9a.75.75 0 101.499-.058l-.346-9zm5.48.058a.75.75 0 10-1.498-.058l-.347 9a.75.75 0 001.5.058l.345-9z" />
        </svg>
	);
}


function PencilIcon({ color = '#000', height = '16', width = '16' }) {
	return (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" height={height} width={width}>
            <path fill={color} d="M21.731 2.269a2.625 2.625 0 00-3.712 0l-1.157 1.157 3.712 3.712 1.157-1.157a2.625 2.625 0 000-3.712zM19.513 8.199l-3.712-3.712-12.15 12.15a5.25 5.25 0 00-1.32 2.214l-.8 2.685a.75.75 0 00.933.933l2.685-.8a5.25 5.25 0 002.214-1.32L19.513 8.2z" />
        </svg>
	);
}





export { XMark, Search, PlusIcon, ThreadsIcon, EngineeringIcon, CheckIcon, TrashIcon, PencilIcon };



