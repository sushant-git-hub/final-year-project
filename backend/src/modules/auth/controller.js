// import { authService } from './service.js';

// export async function login(req, reply) {
//     const { email, password } = req.body;
//     try {
//         const user = await authService.login(email, password);
//         return user;
//     } catch (err) {
//         reply.code(401).send({ error: err.message });
//     }
// }

// export async function register(req, reply) {
//     try {
//         const user = await authService.register(req.body);
//         return user;
//     } catch (err) {
//         reply.code(400).send({ error: err.message });
//     }
// }
