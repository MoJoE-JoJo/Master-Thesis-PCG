package gym;

import engine.core.*;
import engine.helper.Assets;
import engine.helper.GameStatus;
import engine.helper.MarioActions;
import py4j.GatewayServer;

import javax.swing.*;
import java.awt.*;
import java.awt.image.VolatileImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;

public class MarioGym {
    static String level;
    static int gameSeconds;
    static int marioState;
    //static boolean visual;

    //Visualisation
    static JFrame window = null;
    static MarioRender render = null;
    static VolatileImage renderTarget = null;
    static Graphics backBuffer = null;
    static Graphics currentBuffer = null;

    //Game and character
    static MarioWorld world = null;
    static Py4JAgent agent = null;
    static MarioTimer agentTimer = null;

    //GameLoop
    static ArrayList<MarioEvent> gameEvents = null;
    static ArrayList<MarioAgentEvent> agentEvents = null;

    //Step related
    static float rewardPos = 0;
    static int rewardTimePenalty = 0;
    static int rewardDeathPenalty = 0;


    public static void main(String[] args) {
        MarioGym gym = new MarioGym();
        // app is now the gateway.entry_point
        GatewayServer server = new GatewayServer(gym);
        server.start();
        System.out.println("Gateway Started");
    }

    public static StepReturnType step(boolean left, boolean right, boolean down, boolean speed, boolean jump){
        agentInput(left, right, down, speed, jump);
        gameUpdate();
        StepReturnType returnVal = new StepReturnType();
        //Done value
        if (world.gameStatus == GameStatus.RUNNING) returnVal.done = false;
        else returnVal.done = true;
        //Reward value
        returnVal.reward = (int) rewardPos + rewardTimePenalty + rewardDeathPenalty;
        returnVal.reward = Math.max(-15, Math.min(15, returnVal.reward));
        //State value
        returnVal.state = world.getMergedObservation(world.mario.x, world.mario.y, 0, 0);
        //Info values
        returnVal.info = new HashMap<>();
        if(world.gameStatus == GameStatus.WIN) returnVal.info.put("Result", "Win");
        else if (world.gameStatus == GameStatus.LOSE) returnVal.info.put("Result", "Lose");
        returnVal.info.put("Yolo","Swaggins");
        return returnVal;
    }

    public static void init(String paramLevel, String imageDirectory, int timer, int paramMarioState, boolean visual){
        level = paramLevel;
        gameSeconds = timer;
        marioState = paramMarioState;
        Assets.img = imageDirectory;

        if (visual) {
            window = new JFrame("Mario AI Framework");
            render = new MarioRender(2);
            window.setContentPane(render);
            window.pack();
            window.setResizable(false);
            window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            render.init();
            window.setVisible(true);
        }

        reset(visual);
        System.out.println("Gym initialised");
    }

    public static void gameUpdate(){
        if (world.gameStatus == GameStatus.RUNNING) {
            //System.out.println(currentTime);
            //get actions
            agentTimer = new MarioTimer(MarioGame.maxTime);
            boolean[] actions = agent.getActions(new MarioForwardModel(world.clone()), agentTimer);
            if (MarioGame.verbose) {
                if (agentTimer.getRemainingTime() < 0 && Math.abs(agentTimer.getRemainingTime()) > MarioGame.graceTime) {
                    System.out.println("The Agent is slowing down the game by: "
                            + Math.abs(agentTimer.getRemainingTime()) + " msec.");
                }
            }
            //Reward info before update
            int tickBeforeUpdate = world.currentTick;
            float marioXBeforeUpdate = world.mario.x;
            // update world
            world.update(actions);
            gameEvents.addAll(world.lastFrameEvents);
            agentEvents.add(new MarioAgentEvent(actions, world.mario.x,
                    world.mario.y, (world.mario.isLarge ? 1 : 0) + (world.mario.isFire ? 1 : 0),
                    world.mario.onGround, world.currentTick));
            //Reward info after update
            int tickAfterUpdate = world.currentTick;
            float marioXAfterUpdate = world.mario.x;
            //Calculate reward components
            rewardPos = marioXAfterUpdate - marioXBeforeUpdate;
            rewardTimePenalty = tickBeforeUpdate - tickAfterUpdate;
            if(world.gameStatus == GameStatus.LOSE) rewardDeathPenalty = -15;
            else rewardDeathPenalty = 0;
            System.out.println("Postion reward: " + rewardPos + ", Time reward: " + rewardTimePenalty + ", Death reward: " + rewardDeathPenalty);

        }
    }

    public static StepReturnType reset(boolean visual){
        agent = new Py4JAgent();
        world = new MarioWorld(null);

        world.visuals = visual;
        world.initializeLevel(level, 1000 * gameSeconds);
        if (visual) {
            world.initializeVisuals(render.getGraphicsConfiguration());
        }
        world.mario.isLarge = marioState > 0;
        world.mario.isFire = marioState > 1;

        world.update(new boolean[MarioActions.numberOfActions()]);

        //initialize graphics
        renderTarget = null;
        backBuffer = null;
        currentBuffer = null;
        if (visual) {
            renderTarget = render.createVolatileImage(MarioGame.width, MarioGame.height);
            backBuffer = render.getGraphics();
            currentBuffer = renderTarget.getGraphics();
            render.addFocusListener(render); //TODO: Maybe not needed
        }

        agentTimer = new MarioTimer(MarioGame.maxTime);
        agent.initialize(new MarioForwardModel(world.clone()), agentTimer);

        gameEvents = new ArrayList<>();
        agentEvents = new ArrayList<>();

        System.out.println("Gym Reset");

        StepReturnType returnVal = new StepReturnType();
        returnVal.done = false;
        returnVal.reward = 0;
        returnVal.state = world.getMergedObservation(world.mario.x, world.mario.y, 0, 0);
        returnVal.info = new HashMap<>();
        return returnVal;
    }

    public static void render(){
        /*
        if (firstRender) {
            window = new JFrame("Mario AI Framework");
            render = new MarioRender(2);
            window.setContentPane(render);
            window.pack();
            window.setResizable(false);
            window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            render.init();
            window.setVisible(true);
            world.initializeVisuals(render.getGraphicsConfiguration());
            renderTarget = render.createVolatileImage(MarioGame.width, MarioGame.height);
            backBuffer = render.getGraphics();
            currentBuffer = renderTarget.getGraphics();
            render.addFocusListener(render); //TODO: Maybe not needed
            firstRender = false;
        }
        */

        render.renderWorld(world, renderTarget, backBuffer, currentBuffer);
    }

    public static void playGame(String level, int time, int marioState, boolean visuals){
        MarioGame game = new MarioGame();
        agent = new Py4JAgent();
        printResults(game.runGame(agent, level, time, marioState, visuals));
    }

    public static void agentInput(boolean left, boolean right, boolean down, boolean speed, boolean jump){
        boolean[] actions = new boolean[]{left, right, down, speed, jump};
        agent.setActions(actions);
    }

    public static void setLevel(String levelParam){
        level = levelParam;
    }

    private static String getLevel(String filepath) {
        String content = "";
        try {
            content = new String(Files.readAllBytes(Paths.get(filepath)));
        } catch (IOException e) {
        }
        return content;
    }

    private static void printResults(MarioResult result) {
        System.out.println("****************************************************************");
        System.out.println("Game Status: " + result.getGameStatus().toString() +
                " Percentage Completion: " + result.getCompletionPercentage());
        System.out.println("Lives: " + result.getCurrentLives() + " Coins: " + result.getCurrentCoins() +
                " Remaining Time: " + (int) Math.ceil(result.getRemainingTime() / 1000f));
        System.out.println("Mario State: " + result.getMarioMode() +
                " (Mushrooms: " + result.getNumCollectedMushrooms() + " Fire Flowers: " + result.getNumCollectedFireflower() + ")");
        System.out.println("Total Kills: " + result.getKillsTotal() + " (Stomps: " + result.getKillsByStomp() +
                " Fireballs: " + result.getKillsByFire() + " Shells: " + result.getKillsByShell() +
                " Falls: " + result.getKillsByFall() + ")");
        System.out.println("Bricks: " + result.getNumDestroyedBricks() + " Jumps: " + result.getNumJumps() +
                " Max X Jump: " + result.getMaxXJump() + " Max Air Time: " + result.getMaxJumpAirTime());
        System.out.println("****************************************************************");
    }
}
