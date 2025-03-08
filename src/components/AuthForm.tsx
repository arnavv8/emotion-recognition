import React, { useState } from 'react';
import { useAuth } from './AuthProvider';
import { toast } from 'react-hot-toast';

export function AuthForm() {
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [otp, setOtp] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isResetPassword, setIsResetPassword] = useState(false);
  const [resetSent, setResetSent] = useState(false);
  const [isVerifying, setIsVerifying] = useState(false);
  const { signIn, signUp, resetPassword, verifyOtp } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    try {
      if (isResetPassword) {
        await resetPassword(email);
        setResetSent(true);
        toast.success('Password reset instructions sent to your email');
      } else if (isSignUp) {
        if (!isVerifying) {
          await signUp(email, password);
          setIsVerifying(true);
          toast.success('OTP sent to your email');
        } else {
          await verifyOtp(email, otp);
          toast.success('Email verified successfully');
        }
      } else {
        await signIn(email, password);
        toast.success('Signed in successfully');
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Authentication failed';
      setError(message);
      toast.error(message);
    }
  };

  const renderOtpVerification = () => (
    <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
      <div className="rounded-md shadow-sm -space-y-px">
        <div>
          <label htmlFor="otp" className="sr-only">
            OTP Code
          </label>
          <input
            id="otp"
            name="otp"
            type="text"
            required
            value={otp}
            onChange={(e) => setOtp(e.target.value)}
            className="appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
            placeholder="Enter OTP code"
          />
        </div>
      </div>

      <div>
        <button
          type="submit"
          className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
        >
          Verify OTP
        </button>
      </div>
    </form>
  );

  const renderResetPassword = () => (
    <>
      <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
        Reset your password
      </h2>
      {resetSent ? (
        <div className="mt-4 bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded">
          If an account exists with this email, you'll receive password reset instructions.
        </div>
      ) : (
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="rounded-md shadow-sm -space-y-px">
            <div>
              <label htmlFor="email-address" className="sr-only">
                Email address
              </label>
              <input
                id="email-address"
                name="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="appearance-none rounded-md relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
                placeholder="Email address"
              />
            </div>
          </div>

          <div>
            <button
              type="submit"
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              Send reset instructions
            </button>
          </div>

          <div className="text-center">
            <button
              type="button"
              onClick={() => {
                setIsResetPassword(false);
                setResetSent(false);
              }}
              className="text-indigo-600 hover:text-indigo-500"
            >
              Back to sign in
            </button>
          </div>
        </form>
      )}
    </>
  );

  const renderAuthForm = () => (
    <>
      <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
        {isSignUp ? 'Create an account' : 'Sign in to your account'}
      </h2>

      <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
        <div className="rounded-md shadow-sm -space-y-px">
          <div>
            <label htmlFor="email-address" className="sr-only">
              Email address
            </label>
            <input
              id="email-address"
              name="email"
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-t-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
              placeholder="Email address"
            />
          </div>
          <div>
            <label htmlFor="password" className="sr-only">
              Password
            </label>
            <input
              id="password"
              name="password"
              type="password"
              autoComplete="current-password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-b-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
              placeholder="Password"
            />
          </div>
        </div>

        {!isSignUp && (
          <div className="flex items-center justify-end">
            <div className="text-sm">
              <button
                type="button"
                onClick={() => setIsResetPassword(true)}
                className="font-medium text-indigo-600 hover:text-indigo-500"
              >
                Forgot your password?
              </button>
            </div>
          </div>
        )}

        <div>
          <button
            type="submit"
            className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            {isSignUp ? 'Sign up' : 'Sign in'}
          </button>
        </div>

        <div className="text-center">
          <button
            type="button"
            onClick={() => {
              setIsSignUp(!isSignUp);
              setIsVerifying(false);
              setOtp('');
            }}
            className="text-indigo-600 hover:text-indigo-500"
          >
            {isSignUp
              ? 'Already have an account? Sign in'
              : "Don't have an account? Sign up"}
          </button>
        </div>
      </form>
    </>
  );

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8 bg-white p-8 rounded-lg shadow-md">
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            {error}
          </div>
        )}

        {isSignUp && isVerifying ? (
          renderOtpVerification()
        ) : isResetPassword ? (
          renderResetPassword()
        ) : (
          renderAuthForm()
        )}
      </div>
    </div>
  );
}