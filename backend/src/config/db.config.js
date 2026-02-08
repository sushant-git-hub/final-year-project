export const dbConfig = {
    host: process.env.DB_HOST || 'localhost',
    port: Number(process.env.DB_PORT) || 5432,
    user: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD || 'password',
    database: process.env.DB_NAME || 'mapmystore',
    max: 20, // max connection pool size
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 2000,
};
