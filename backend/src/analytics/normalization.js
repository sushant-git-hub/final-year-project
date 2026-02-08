/**
 * Normalize a value between 0 and 1 using min-max normalization
 * @param {number} value - The value to normalize
 * @param {number} min - Minimum expected value
 * @param {number} max - Maximum expected value
 * @returns {number} Normalized value between 0 and 1
 */
export function normalizeMinMax(value, min, max) {
    if (max === min) return 1;
    const normalized = (value - min) / (max - min);
    return Math.max(0, Math.min(1, normalized)); // Clamp between 0 and 1
}

/**
 * Standardize an array of numbers (Z-score normalization)
 * @param {number[]} data - Array of numbers
 * @returns {number[]} Standardized values
 */
export function standardizeZScore(data) {
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const stdDev = Math.sqrt(
        data.map((x) => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) /
        data.length
    );

    if (stdDev === 0) return data.map(() => 0);
    return data.map((x) => (x - mean) / stdDev);
}
