// Mock Data based on provided summary (since we don't have the CSV file yet)
const mockWards = [
    { wardId: 'PMC_1', name: 'Aundh', population: 45000, lat: 18.5580, lng: 73.8075 },
    { wardId: 'PMC_2', name: 'Baner', population: 52000, lat: 18.5642, lng: 73.7769 },
    { wardId: 'PMC_3', name: 'Kothrud', population: 78000, lat: 18.5074, lng: 73.8077 },
    { wardId: 'PMC_4', name: 'Shivajinagar', population: 35000, lat: 18.5314, lng: 73.8446 },
    { wardId: 'PMC_5', name: 'Viman Nagar', population: 42000, lat: 18.5679, lng: 73.9143 },
    { wardId: 'PCMC_1', name: 'Wakad', population: 65000, lat: 18.5987, lng: 73.7667 },
    { wardId: 'PCMC_2', name: 'Pimple Saudagar', population: 58000, lat: 18.5991, lng: 73.7925 },
];

/**
 * Repository to handle demographic data
 */
export const demographicRepository = {
    /**
     * Find the nearest ward to a given location (lat, lng)
     * @param {number} lat
     * @param {number} lng
     * @returns {Object|null} Nearest ward object or null
     */
    getNearestWard(lat, lng) {
        let nearest = null;
        let minDistance = Infinity;

        for (const ward of mockWards) {
            const dist = Math.sqrt(
                Math.pow(ward.lat - lat, 2) + Math.pow(ward.lng - lng, 2)
            );
            if (dist < minDistance) {
                minDistance = dist;
                nearest = ward;
            }
        }

        // Threshold check (optional, e.g., if > 0.1 degrees away, maybe no data)
        return nearest;
    },

    getAllWards() {
        return mockWards;
    }
};
