import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:titweng/models/user_model.dart';

class FirebaseAuthService {
  static final FirebaseAuthService _instance = FirebaseAuthService._internal();
  factory FirebaseAuthService() => _instance;
  FirebaseAuthService._internal();

  final FirebaseAuth _auth = FirebaseAuth.instance;
  final GoogleSignIn _googleSignIn = GoogleSignIn();
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  User? get currentUser => _auth.currentUser;
  Stream<User?> get authStateChanges => _auth.authStateChanges();

  // Email/Password Sign Up
  Future<void> signUpWithEmail({
    required String email,
    required String password,
    required String name,
  }) async {
    try {
      final credential = await _auth.createUserWithEmailAndPassword(
        email: email,
        password: password,
      );

      if (credential.user != null) {
        // Update display name
        await credential.user!.updateDisplayName(name);
        
        // Email verification disabled due to template issues
        // try {
        //   await credential.user!.sendEmailVerification();
        //   print('Verification email sent to: ${credential.user!.email}');
        // } catch (emailError) {
        //   print('Email verification error: $emailError');
        // }
      }
    } on FirebaseAuthException catch (e) {
      throw _handleAuthException(e);
    }
  }

  // Email/Password Sign In
  Future<UserModel?> signInWithEmail({
    required String email,
    required String password,
  }) async {
    try {
      final credential = await _auth.signInWithEmailAndPassword(
        email: email,
        password: password,
      );

      if (credential.user != null) {
        // Email verification disabled - users can sign in without verification
        // if (!credential.user!.emailVerified) {
        //   throw 'Please check your email and click the verification link we sent you before signing in.';
        // }
        
        // Create user model directly from Firebase Auth
        final userModel = UserModel(
          id: credential.user!.uid,
          name: credential.user!.displayName ?? 'User',
          email: email,
          phone: '',
          role: UserRole.user,
          createdAt: DateTime.now(),
        );
        return userModel;
      }
    } on FirebaseAuthException catch (e) {
      throw _handleAuthException(e);
    }
    return null;
  }

  // Google Sign In
  Future<UserModel?> signInWithGoogle() async {
    try {
      final GoogleSignInAccount? googleUser = await _googleSignIn.signIn();
      if (googleUser == null) return null;

      final GoogleSignInAuthentication googleAuth = await googleUser.authentication;
      final credential = GoogleAuthProvider.credential(
        accessToken: googleAuth.accessToken,
        idToken: googleAuth.idToken,
      );

      final userCredential = await _auth.signInWithCredential(credential);
      
      if (userCredential.user != null) {
        final user = userCredential.user!;
        
        // Create UserModel directly from Firebase Auth user
        final userModel = UserModel(
          id: user.uid,
          name: user.displayName ?? 'Google User',
          email: user.email ?? '',
          phone: '',
          role: UserRole.user,
          createdAt: DateTime.now(),
        );

        // Skip Firestore for Google Sign-In to avoid offline issues

        return userModel;
      }
    } catch (e) {
      throw 'Google sign-in failed: ${e.toString()}';
    }
    return null;
  }

  // Password Reset
  Future<void> resetPassword(String email) async {
    try {
      await _auth.sendPasswordResetEmail(email: email);
    } on FirebaseAuthException catch (e) {
      throw _handleAuthException(e);
    }
  }

  // Get user with role from Firestore
  Future<UserModel?> getUserWithRole() async {
    final user = _auth.currentUser;
    if (user == null) return null;

    try {
      final doc = await _firestore.collection('users').doc(user.uid).get();
      
      if (doc.exists) {
        return UserModel.fromJson(doc.data()!);
      }
    } catch (e) {
      print('Error getting user role: $e');
    }
    return null;
  }

  // Sign Out
  Future<void> signOut() async {
    await Future.wait([
      _auth.signOut(),
      _googleSignIn.signOut(),
    ]);
  }

  // Get user from Firestore
  Future<UserModel?> _getUserFromFirestore(String uid) async {
    try {
      final doc = await _firestore.collection('users').doc(uid).get();
      if (doc.exists) {
        final data = doc.data()!;
        return UserModel(
          id: uid,
          name: data['name'] ?? '',
          email: data['email'] ?? '',
          phone: data['phone'] ?? '',
          role: UserRole.user,
          createdAt: data['createdAt'] != null ? (data['createdAt'] as Timestamp).toDate() : DateTime.now(),
        );
      }
    } catch (e) {
      print('Error getting user from Firestore: $e');
    }
    return null;
  }

  // Handle Firebase Auth exceptions
  String _handleAuthException(FirebaseAuthException e) {
    switch (e.code) {
      case 'weak-password':
        return 'Please choose a stronger password with at least 6 characters.';
      case 'email-already-in-use':
        return 'This email is already registered. Try signing in instead.';
      case 'user-not-found':
        return 'We couldn\'t find an account with this email. Would you like to create a new account?';
      case 'wrong-password':
        return 'The password you entered is incorrect. Please try again.';
      case 'invalid-email':
        return 'Please enter a valid email address.';
      case 'invalid-credential':
        return 'The email or password you entered is incorrect.';
      case 'user-disabled':
        return 'This account has been temporarily disabled. Please contact support.';
      case 'too-many-requests':
        return 'Too many failed attempts. Please wait a moment and try again.';
      case 'network-request-failed':
        return 'Please check your internet connection and try again.';
      default:
        return 'Something went wrong. Please try again.';
    }
  }
}