// Run this in Firebase Console → Functions or locally with Admin SDK
const admin = require('firebase-admin');

// Initialize Firebase Admin (if not already done)
// admin.initializeApp();

async function createAdminUser() {
  try {
    // Create user in Authentication
    const userRecord = await admin.auth().createUser({
      email: 'admin@titweng.com',
      password: 'Admin123!',
      displayName: '[ADMIN] Admin User',
      emailVerified: true
    });

    // Set custom claims
    await admin.auth().setCustomUserClaims(userRecord.uid, {
      role: 'admin',
      isAdmin: true
    });

    // Create Firestore document
    await admin.firestore().collection('users').doc(userRecord.uid).set({
      id: userRecord.uid,
      name: 'Admin User',
      email: 'admin@titweng.com',
      phone: '+211-XXX-XXXX',
      role: 'admin',
      isAdmin: true,
      created_at: new Date().toISOString()
    });

    console.log('Admin user created successfully:', userRecord.uid);
  } catch (error) {
    console.error('Error creating admin user:', error);
  }
}

createAdminUser();