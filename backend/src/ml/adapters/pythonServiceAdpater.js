import axios from 'axios';
import { MLAdapter } from './mlAdapter.js';

export class PythonServiceAdapter extends MLAdapter {
  async predict(features) {
    const res = await axios.post(
      process.env.ML_SERVICE_URL,
      { features }
    );

    return res.data.predictions;
  }
}