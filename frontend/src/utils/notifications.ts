export interface NotificationOptions {
  title: string;
  body: string;
  icon?: string;
  badge?: string;
  requireInteraction?: boolean;
  silent?: boolean;
  autoClose?: number; // time in milliseconds
  tag?: string;
}

export class NotificationManager {
  static isSupported(): boolean {
    return 'Notification' in window;
  }

  static getPermission(): NotificationPermission {
    if (!this.isSupported()) {
      throw new Error('Notifications are not supported by the browser');
    }
    return Notification.permission;
  }

  static async requestPermission(): Promise<NotificationPermission> {
    if (!this.isSupported()) {
      throw new Error('Notifications are not supported by the browser');
    }

    console.log('🔔 Requesting notification permission...');
    
    try {
      const permission = await Notification.requestPermission();
      console.log('📝 Permission received:', permission);
      
      if (permission === 'granted') {
        console.log('✅ Permissions granted');
      } else if (permission === 'denied') {
        console.log('❌ Permissions denied by user');
        console.log('🛠️ Solution: Browser settings → Notifications → Allow for this site');
      }
      
      return permission;
    } catch (error) {
      console.error('❌ Error requesting permission:', error);
      throw error;
    }
  }

  static async show(options: NotificationOptions): Promise<Notification | null> {
    console.log('🔔 Attempting to send notification:', options.title);

    if (!this.isSupported()) {
      console.warn('⚠️ Notifications are not supported by the browser');
      return null;
    }

    let permission = this.getPermission();
    console.log('📋 Current permission:', permission);
    
    if (permission === 'default') {
      console.log('❓ Permission not granted, requesting...');
      permission = await this.requestPermission();
    }

    if (permission !== 'granted') {
      console.warn('⚠️ Notifications are blocked by user:', permission);
      console.log('🛠️ How to fix: Browser settings → Notifications → Allow');
      return null;
    }

    try {
      console.log('🛠️ Creating notification...');
      console.log('📋 Options:', {
        title: options.title,
        body: options.body,
        icon: options.icon,
        requireInteraction: options.requireInteraction,
        silent: options.silent,
        tag: options.tag
      });
      
      const startTime = Date.now();
      let systemNotificationShown = false;
      
      const notification = new Notification(options.title, {
        body: options.body,
        icon: options.icon || '/favicon.ico',
        badge: options.badge || '/favicon.ico',
        requireInteraction: options.requireInteraction ?? true,
        silent: options.silent ?? false,
        tag: options.tag || 'default'
      });

      // Detailed event logging
      notification.onshow = () => {
        systemNotificationShown = true;
        const delay = Date.now() - startTime;
        console.log(`✅ SUCCESS! System notification shown after ${delay}ms`);
      };

      notification.onclick = () => {
        console.log('👆 User clicked notification');
        window.focus();
        notification.close();
      };

      notification.onclose = () => {
        console.log('📭 Notification closed');
      };

      notification.onerror = (error) => {
        console.error('❌ Notification error:', error);
      };

      // Check if system notification appeared
      setTimeout(() => {
        if (!systemNotificationShown) {
          console.warn('🚨 WARNING: System notification did NOT appear!');
          console.log('🔍 Possible reasons:');
          console.log('   1. 🍎 macOS: System Settings → Notifications & Focus → [Browser] → Allow');
          console.log('   2. 😴 "Do Not Disturb" mode is active');
          console.log('   3. 🕵️ Private browsing mode');
          console.log('   4. ⚙️ Specific browser settings');
          console.log('💡 Try: System Settings → Notifications & Focus → Chrome → Enable notifications');
        }
      }, 2000);

      // Auto close
      if (options.autoClose && options.autoClose > 0) {
        console.log(`⏰ Notification will close in ${options.autoClose}ms`);
        setTimeout(() => {
          console.log('⏰ Auto-closing notification');
          notification.close();
        }, options.autoClose);
      }

      console.log('✅ Notification created and sent, waiting for onshow event...');
      return notification;
      
    } catch (error) {
      console.error('❌ Error creating notification:', error);
      throw error;
    }
  }

  static async showPodcastComplete(): Promise<Notification | null> {
    console.log('🎙️ Sending podcast ready notification...');
    
    return this.show({
      title: '🎙️ Podcast ready!',
      body: 'Podcast generation completed successfully. Click to view.',
      icon: '/favicon.ico',
      requireInteraction: true,
      silent: false,
      autoClose: 10000, // 10 seconds
      tag: 'podcast-complete'
    });
  }

  static async showWelcome(): Promise<Notification | null> {
    console.log('👋 Sending welcome notification...');
    
    return this.show({
      title: '👋 Welcome!',
      body: 'Notifications set up successfully',
      icon: '/favicon.ico',
      requireInteraction: false,
      silent: true,
      autoClose: 3000, // 3 seconds
      tag: 'welcome'
    });
  }

  static async testNotification(): Promise<Notification | null> {
    console.log('🧪 Sending test notification...');
    
    return this.show({
      title: '🧪 Test notification',
      body: 'If you see this on your desktop, push notifications are working!',
      icon: '/favicon.ico',
      requireInteraction: false,
      silent: false,
      autoClose: 8000, // 8 seconds
      tag: 'test'
    });
  }

  // Method for diagnostics
  static diagnose(): void {
    console.log('=== 🔍 PUSH NOTIFICATION DIAGNOSTICS ===');
    
    console.log('🌐 API support:', this.isSupported() ? '✅ Yes' : '❌ No');
    
    if (this.isSupported()) {
      console.log('📋 Permission:', this.getPermission());
      console.log('👤 User Agent:', navigator.userAgent);
      console.log('👁️ Document visibility:', document.visibilityState);
      console.log('🎯 Window focus:', document.hasFocus());
      console.log('🔗 Protocol:', window.location.protocol);
      console.log('🏠 Host:', window.location.host);
      console.log('🖥️ Platform:', navigator.platform);
      
      if (navigator.platform.toLowerCase().includes('mac')) {
        console.log('🍎 macOS detected');
        console.log('💡 IMPORTANT: Check System Settings → Notifications & Focus → [Your browser]');
        console.log('💡 IMPORTANT: Make sure "Do Not Disturb" is off');
      }
    }
    
    console.log('=== 🔍 END OF DIAGNOSTICS ===');
  }
} 