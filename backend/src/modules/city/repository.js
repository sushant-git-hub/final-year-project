const cities = {
  delhi: {
    id: 'delhi',
    name: 'Delhi',
    hexResolution: 8,
    boundary: [
      [
        [77.1025, 28.7041],
        [77.3450, 28.7041],
        [77.3450, 28.4041],
        [77.1025, 28.4041],
        [77.1025, 28.7041]
      ]
    ]
  },
  pune: {
    id: 'pune',
    name: 'Pune',
    hexResolution: 8,
    boundary: [
      [
        [73.7400, 18.6200], // NW
        [74.0200, 18.6200], // NE
        [74.0200, 18.4400], // SE
        [73.7400, 18.4400], // SW
        [73.7400, 18.6200]  // NW (Close loop)
      ]
    ]
  }
};

export const cityRepository = {
  async getCityByName(name) {
    return cities[name?.toLowerCase()];
  }
};