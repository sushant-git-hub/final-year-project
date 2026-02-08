export const SCORING_WEIGHTS = {
    DEFAULT: {
        demand: 0.45,
        coverage: 0.30,
        cost: 0.25,
    },
    RETAIL: {
        demand: 0.50,
        coverage: 0.30,
        cost: 0.20,
    },
    LOGISTICS: {
        demand: 0.20,
        coverage: 0.40,
        cost: 0.40,
    },
};

export const THRESHOLDS = {
    MAX_DELIVERY_RADIUS_KM: 20,
    MIN_DEMAND_SCORE: 0.1,
};
