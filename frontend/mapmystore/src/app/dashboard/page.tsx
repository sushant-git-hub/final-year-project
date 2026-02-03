import DashboardLayout from "@/components/Dashboard/DashboardLayout";
import StoreForm from "@/components/StoreForm";
// import MapSection from "@/components/MapSection";
// import ChartsSection from "@/components/ChartsSection";
// import ResultsTable from "@/components/ResultsTable";

export default function DashboardPage() {
  return (
    <>
    <DashboardLayout>
      <h1 className="mb-4 text-2xl font-bold">
        Dashboard
      </h1>
        <StoreForm />
        
    {/* <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <MapSection />
        <ChartsSection />
      </div>

      <div className="mt-6">
        <ResultsTable />
      </div> */}
    </DashboardLayout>
      </>
  );
}
