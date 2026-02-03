import requests
import json
import csv
import time
from typing import List, Dict, Set

class EnhancedPuneRetailFinder:
    def __init__(self, api_key: str):
        """
        Enhanced retail store finder that searches Pune comprehensively.
        Uses grid search + area-based search to find ALL stores.
        """
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/place"
        
        # Comprehensive list of Pune areas
        self.pune_areas = [
            "Kothrud", "Hinjewadi", "Wakad", "Pimple Saudagar", "Baner",
            "Aundh", "Shivajinagar", "Deccan", "Sadashiv Peth", "FC Road",
            "Camp", "Koregaon Park", "Kalyani Nagar", "Viman Nagar", "Kharadi",
            "Hadapsar", "Magarpatta", "Wanowrie", "Kondhwa", "Undri",
            "Katraj", "Sinhagad Road", "Warje", "Karve Nagar", "Kothrud",
            "Paud Road", "Bavdhan", "Sus", "Pashan", "Pune University",
            "Model Colony", "Parvati", "Bibwewadi", "Dhankawadi", "Sahakarnagar",
            "Khadki", "Pimpri", "Chinchwad", "Nigdi", "Akurdi",
            "Bhosari", "Chakan", "Talegaon", "Ravet", "Tathawade",
            "Moshi", "Alandi Road", "Vishrantwadi", "Dhanori", "Wagholi",
            "Lohegaon", "Yerawada", "Ghorpadi", "Mundhwa", "Fursungi",
            "Handewadi", "Manjri", "Phursungi", "Pisoli", "Saswad Road"
        ]
        
        # Grid coordinates covering Greater Pune (60+ zones)
        self.grid_points = self._generate_grid()
        
        # Store types to search
        self.store_types = [
            'store', 'clothing_store', 'shoe_store', 'jewelry_store',
            'electronics_store', 'furniture_store', 'home_goods_store',
            'book_store', 'supermarket', 'shopping_mall', 'convenience_store',
            'department_store', 'hardware_store', 'pet_store', 'liquor_store',
            'bicycle_store', 'florist', 'gift_shop', 'toy_store', 'bakery',
            'grocery_or_supermarket', 'pharmacy', 'beauty_salon', 'hair_care',
            'spa', 'gym', 'restaurant', 'cafe', 'bar'
        ]
        
        self.all_stores = []
        self.seen_place_ids: Set[str] = set()
        
    def _generate_grid(self) -> List[Dict]:
        """
        Generate a grid of coordinates covering Pune metropolitan area.
        Creates a 8x8 grid (64 points) for comprehensive coverage.
        """
        # Pune boundaries (approximate)
        lat_min, lat_max = 18.4, 18.7  # North-South
        lng_min, lng_max = 73.7, 73.95  # East-West
        
        grid_size = 8  # 8x8 = 64 search zones
        grid_points = []
        
        lat_step = (lat_max - lat_min) / grid_size
        lng_step = (lng_max - lng_min) / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                lat = lat_min + (i * lat_step) + (lat_step / 2)
                lng = lng_min + (j * lng_step) + (lng_step / 2)
                grid_points.append({'lat': lat, 'lng': lng})
        
        return grid_points
    
    def search_nearby(self, lat: float, lng: float, store_type: str, radius: int = 5000) -> List[Dict]:
        """
        Search nearby stores at specific coordinates.
        Smaller radius (5km) for better coverage.
        """
        url = f"{self.base_url}/nearbysearch/json"
        
        params = {
            'location': f"{lat},{lng}",
            'radius': radius,
            'type': store_type,
            'key': self.api_key
        }
        
        stores = []
        page_count = 0
        
        while page_count < 3:  # Max 3 pages (60 results)
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                break
            
            data = response.json()
            
            if data['status'] == 'OK':
                for store in data['results']:
                    if store['place_id'] not in self.seen_place_ids:
                        self.seen_place_ids.add(store['place_id'])
                        stores.append(store)
                
                if 'next_page_token' in data:
                    params = {
                        'pagetoken': data['next_page_token'],
                        'key': self.api_key
                    }
                    time.sleep(2)
                    page_count += 1
                else:
                    break
            else:
                break
        
        return stores
    
    def search_text(self, query: str) -> List[Dict]:
        """
        Text-based search for stores in specific areas.
        More flexible than nearby search.
        """
        url = f"{self.base_url}/textsearch/json"
        
        params = {
            'query': query,
            'key': self.api_key
        }
        
        stores = []
        page_count = 0
        
        while page_count < 3:
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                break
            
            data = response.json()
            
            if data['status'] == 'OK':
                for store in data['results']:
                    if store['place_id'] not in self.seen_place_ids:
                        self.seen_place_ids.add(store['place_id'])
                        stores.append(store)
                
                if 'next_page_token' in data:
                    params = {
                        'pagetoken': data['next_page_token'],
                        'key': self.api_key
                    }
                    time.sleep(2)
                    page_count += 1
                else:
                    break
            else:
                break
        
        return stores
    
    def comprehensive_search(self):
        """
        Perform comprehensive search using multiple strategies.
        """
        total_searches = 0
        
        print("=" * 70)
        print("STRATEGY 1: Grid-based Nearby Search")
        print("=" * 70)
        
        # Strategy 1: Grid search with high-priority store types
        priority_types = ['store', 'shopping_mall', 'supermarket', 'clothing_store', 'electronics_store']
        
        for idx, point in enumerate(self.grid_points, 1):
            print(f"\nGrid Point {idx}/{len(self.grid_points)} - Lat: {point['lat']:.4f}, Lng: {point['lng']:.4f}")
            
            for store_type in priority_types:
                stores = self.search_nearby(point['lat'], point['lng'], store_type, radius=3000)
                if stores:
                    self.all_stores.extend(stores)
                    print(f"  {store_type}: +{len(stores)} stores (Total unique: {len(self.seen_place_ids)})")
                
                total_searches += 1
                time.sleep(0.5)  # Rate limiting
        
        print("\n" + "=" * 70)
        print("STRATEGY 2: Area-based Text Search")
        print("=" * 70)
        
        # Strategy 2: Text search by area name
        search_terms = ['retail store', 'shop', 'shopping', 'market', 'mall', 'store']
        
        for idx, area in enumerate(self.pune_areas, 1):
            print(f"\nArea {idx}/{len(self.pune_areas)}: {area}")
            
            for term in search_terms[:3]:  # Use top 3 terms to save API calls
                query = f"{term} in {area}, Pune"
                stores = self.search_text(query)
                if stores:
                    self.all_stores.extend(stores)
                    print(f"  '{term}': +{len(stores)} stores (Total unique: {len(self.seen_place_ids)})")
                
                total_searches += 1
                time.sleep(0.5)
        
        print("\n" + "=" * 70)
        print("STRATEGY 3: Additional Store Types")
        print("=" * 70)
        
        # Strategy 3: Comprehensive store type search (central Pune)
        central_pune = {'lat': 18.5204, 'lng': 73.8567}
        
        for idx, store_type in enumerate(self.store_types, 1):
            stores = self.search_nearby(central_pune['lat'], central_pune['lng'], store_type, radius=15000)
            if stores:
                self.all_stores.extend(stores)
                print(f"{idx}. {store_type}: +{len(stores)} stores (Total unique: {len(self.seen_place_ids)})")
            
            total_searches += 1
            time.sleep(0.5)
        
        print("\n" + "=" * 70)
        print(f"Search Complete!")
        print(f"Total API calls made: {total_searches}")
        print(f"Total unique stores found: {len(self.seen_place_ids)}")
        print("=" * 70)
    
    def get_place_details(self, place_id: str) -> Dict:
        """Get detailed information about a place."""
        url = f"{self.base_url}/details/json"
        
        params = {
            'place_id': place_id,
            'fields': 'name,formatted_address,formatted_phone_number,website,opening_hours,rating,user_ratings_total,price_level,types,business_status',
            'key': self.api_key
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'OK':
                return data['result']
        
        return {}
    
    def process_stores(self, get_details: bool = True) -> List[Dict]:
        """
        Process all stores and optionally get detailed information.
        Set get_details=False to skip details and save API calls.
        """
        processed = []
        total = len(self.all_stores)
        
        print(f"\nProcessing {total} stores...")
        
        for idx, store in enumerate(self.all_stores, 1):
            if idx % 100 == 0:
                print(f"Processed {idx}/{total} stores...")
            
            store_info = {
                'place_id': store.get('place_id', ''),
                'name': store.get('name', ''),
                'latitude': store['geometry']['location']['lat'],
                'longitude': store['geometry']['location']['lng'],
                'address': store.get('vicinity', ''),
                'types': ', '.join(store.get('types', [])),
                'rating': store.get('rating', 'N/A'),
                'user_ratings_total': store.get('user_ratings_total', 0),
                'business_status': store.get('business_status', ''),
            }
            
            # Optionally get detailed info (uses more API calls)
            if get_details and idx <= 1000:  # Get details for first 1000 only
                details = self.get_place_details(store['place_id'])
                store_info.update({
                    'formatted_address': details.get('formatted_address', ''),
                    'phone': details.get('formatted_phone_number', ''),
                    'website': details.get('website', ''),
                    'price_level': details.get('price_level', 'N/A'),
                })
                time.sleep(0.1)
            
            processed.append(store_info)
        
        return processed
    
    def save_to_csv(self, stores: List[Dict], filename: str = 'pune_all_retail_stores.csv'):
        """Save to CSV."""
        if not stores:
            print("No stores to save")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=stores[0].keys())
            writer.writeheader()
            writer.writerows(stores)
        
        print(f"\n✅ Data saved to {filename}")
    
    def save_to_json(self, stores: List[Dict], filename: str = 'pune_all_retail_stores.json'):
        """Save to JSON."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(stores, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Data saved to {filename}")


def main():
    """Main function."""
    # Replace with your API key
    API_KEY = "YOUR_GOOGLE_PLACES_API_KEY_HERE"
    
    if API_KEY == "YOUR_GOOGLE_PLACES_API_KEY_HERE":
        print("❌ ERROR: Please add your actual API key")
        return
    
    print("\n" + "=" * 70)
    print("🏪 ENHANCED PUNE RETAIL STORE FINDER")
    print("=" * 70)
    print("\nThis script will:")
    print("✓ Search 64 grid zones across Pune")
    print("✓ Search 50+ area names (Kothrud, Hinjewadi, etc.)")
    print("✓ Use multiple search strategies")
    print("✓ Find 5,000-15,000+ stores")
    print("\n⚠️  This will take 30-60 minutes and use ~$20-50 in API calls")
    print("=" * 70)
    
    proceed = input("\nProceed? (yes/no): ").lower()
    if proceed != 'yes':
        print("Cancelled.")
        return
    
    print("\n🚀 Starting comprehensive search...\n")
    
    finder = EnhancedPuneRetailFinder(API_KEY)
    
    # Perform comprehensive search
    finder.comprehensive_search()
    
    # Process stores (set get_details=False to skip details and save API calls)
    print("\nWould you like detailed info (phone, website) for all stores?")
    print("⚠️  Getting details adds ~$10-20 in API costs")
    get_details = input("Get details? (yes/no): ").lower() == 'yes'
    
    processed = finder.process_stores(get_details=get_details)
    
    # Save results
    finder.save_to_csv(processed)
    finder.save_to_json(processed)
    
    print("\n" + "=" * 70)
    print("🎉 PROCESS COMPLETED!")
    print(f"📊 Total unique stores found: {len(processed)}")
    print("=" * 70)


if __name__ == "__main__":
    main()