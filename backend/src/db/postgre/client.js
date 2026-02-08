// import pg from 'pg';
// import { dbConfig } from '../../config/db.config.js';

// const { Pool } = pg;

// // Create a new pool using the config
// const pool = new Pool(dbConfig);

// pool.on('error', (err) => {
//     console.error('Unexpected error on idle client', err);
//     process.exit(-1);
// });

// export const dbRequest = async (text, params) => {
//     const client = await pool.connect();
//     try {
//         const res = await client.query(text, params);
//         return res;
//     } finally {
//         client.release();
//     }
// };

// export const getClient = () => pool.connect();
