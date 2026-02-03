export default function ExportPanel() {
  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-4">
      {/* Info */}
      <div className="text-sm text-gray-600">
        Export analysis results
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        <button
          className="flex items-center gap-2 rounded bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700"
        >
          📄 Export PDF
        </button>

        <button
          className="flex items-center gap-2 rounded bg-green-600 px-4 py-2 text-sm font-medium text-white hover:bg-green-700"
        >
          📊 Export Excel
        </button>
      </div>
    </div>
  );
}
